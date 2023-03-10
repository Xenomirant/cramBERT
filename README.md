# cramBERT
This repository contains my implementation of the BERT architecture and training, including the data pipeline to train on the OpenWebText2 dataset. I was inspired to carry out this project by recent work on efficient, low-budget BERT training. I came across the "Cramming paper" ([Cramming: Training a Language Model on a Single GPU in One Day](https://arxiv.org/abs/2212.14034)), and was impressed how much performance they could get out of a BERT that they trained for just 24 hours on low-end hardware. They also managed to test lots of trendy architecture and training modifications along the way, and so their paper is the perfect guide to training a BERT with all the bells and whistles. I took this paper as my starting point to train a BERT that I can call my very own! In my case, I'm training on a single A100 in Google Colab, which is a bit nicer and faster than what the authors had access to, but it supports a similar sort of minimalist, scarcity-mindset training setup.

## Current Progress (3/1/2023)
So far, I've managed to train BERT with the MLM loss objective to a reasonable loss of around 1.7 with 10 billion tokens, which is solid (slightly better loss than the Cramming paper, but the loss shouldn't be directly compared since I use a different tokenization scheme.)

## Data and Preprocessing
I train and validate the model using the OpenWebText2 dataset, an open-source reproduction of OpenAI's WebText dataset by EleutherAI, which is a subset of the larger Pile dataset. The Cramming authors experiment with several corpuses, including BookCorpus-Wikipedia, C4, and The Pile—I decided to stick with just one solid dataset. OpenWebText2 comprises over 17 million scraped documents, totaling around 65GB of uncompressed text. The dataset and code are available [here](https://github.com/EleutherAI/openwebtext2), along with preprocessing utilities. I directly borrowed some of this code to load the dataset from JSONL archives. It is licensed under the [MIT License](https://github.com/EleutherAI/openwebtext2/blob/master/LICENSE).

OpenWebText2 is already de-duplicated, so the only additional preprocessing steps I perform are filtering out documents that aren't in English, and filtering out documents that don't compress well when tokenized, as suggested by the Cramming paper. I experimented with some of the filtering techniques suggested by the creaters of the DeepMind Gopher model ([Rae et al., 2022](https://arxiv.org/abs/2112.11446)), but I found that they didn't remove many documents. I suspect that the full Common Crawl scrapes in Gopher's MassiveWeb dataset contain more junk than OpenWebText2, which curates links from Reddit submissions. 

Unlike the authors of the Cramming paper, I do not convert the text to lowercase or strip out special characters and accents before tokenization. Every document from the OpenWebText2 dataset is left as-is.

## Tokenization
I further depart from the Cramming paper in my decision to use a byte-level BPE tokenizer, as I want the model to be able to represent any text on the web, including the special characters 🥺 and accents I chose not to strip out. I use the HuggingFace Tokenizers library to train a BPE tokenizer from scratch on OpenWebText2, filtered to only English webpages. In the pre-tokenization step, the tokenizer applies NFC normalization, adds a prefix space to the start of each document (so that a word at the start of the document is considered the same as a word in the middle of a document), and splits on whitespace using the same regular expression as the GPT2 tokenizer. I use a vocabulary size of 32,768, the same size as the WordPiece vocabulary in the Cramming paper (and a [nice number that makes GPUs happy](https://twitter.com/karpathy/status/1621578354024677377)). After tokenization, all resulting tokens are packed into sequences of length 128, with documents separated by a special `[SEP]` token. This means no padding is required, and no computation is wasted on truncated sequences.

## Model Architecture
My implementation of BERT is quite similar to the [original paper]([https://arxiv](https://arxiv.org/abs/1810.04805)) by Devlin et al., with some tweaks suggested by more recent research. Some of these are identified in the Cramming paper, and many are now commonplace in the most recent wave of Transformer models.

* I use the same basic 12-layer BERT-base setup as the original BERT paper, with 768-dimensional embeddings and 12 attention heads. As is now common, I place LayerNorm modules before, rather than after, each attention and feed-forward sublayer, which improves training stability.
* For simplicity, I use learned absolute position embeddings. This means my model will not generalize beyond the sequence length used for training (128 tokens), but recent work on positional encoding (e.g. [Press, Smith, & Lewis, 2021](https://arxiv.org/abs/2108.12409)) finds that sinusoidal embeddings don't generalize well to longer sequences either.
* The feed-forward networks in my Transformer use the [Gated Linear Units](https://arxiv.org/abs/2002.05202) proposed by Noam Shazeer (2020). Following this paper, I reduce the feed-forward hidden size to 2,048 (rather than 3,072) to maintain the same number of parameters.
* I omit biases for all feed-forward layers, including the query-key-value projections in the attention sublayer. I also omit the bias in the affine transformations that follow LayerNorms. Omitting bias is a common practice in recent Transformer models, and is suggested in the Cramming paper as a way to simplify and speed up training, without substantially reducing the *size* of the model (which tends to hurt performance).
* Weights in all linear layers are initialized randomly from a normal distribution with mean 0 and standard deviation 0.002. I found that a standard deviation of 0.02 (the default value for the Cramming paper and also for OpenAI's [GPT-2](https://github.com/openai/gpt-2/blob/master/src/model.py)) resulted in a large initial loss, indicating that a smaller initialization may work better. I'm sure that Kaiming or Xavier uniform initialization would work fine too, the important thing seems to be making sure the weights are small enough. Positional embeddings were initialized to 0, and the LayerNorm weights were initialized to 1.
* For token embeddings, I use the StableEmbedding module from the `bitsandbytes` library, which is a drop-in replacement for `torch.nn.Embedding` that is more stable when using an 8-bit optimizer. It includes a LayerNorm, so I do not need to add my own LayerNorm directly after the token embedding. I add an additional LayerNorm after summing the positional embedding with the token embedding.

## Training
I drop the next-sentence prediction objective from the original BERT paper, and train only on masked language modeling (MLM), following the Cramming paper. This works fine, and seems to be common nowadays when training a BERT. Rather than a "budget" based on wall-clock time, I limit myself to training on around 10 billion tokens from OpenWebText2 (a similar number of tokens to what the Cramming authors got through in 24 hours). There are a number of helpful tricks I take advantage of to speed up training and make it work on 1 GPU. Full training details:
* I use a short sequence length of 128, which saves a great deal of memory and computation, as attention is quadratic in sequence length.
* I use gradient accumulation to increase the effective batch size. I can fit 256 on an A100 GPU, so that's the micro-batch size. These batches are accumulated up to the maximum batch size of 4,096 sequences (suggested by the Cramming paper and similar work on low-budged BERT training by [Iszak et al., 2021](https://arxiv.org/abs/2104.07705)).
* I "ramp up" the batch size throughout training, as suggested by the Cramming paper, via a simple linear schedule. This is thought to be helpful because it lets the model make more progress faster early in training, when it's learning the "easy" things. Research from OpenAI suggests that the [gradient noise scale](https://openai.com/blog/science-of-ai/) grows large later in training, when the model is learning more "difficult" things, so it probably helps to have a larger batch size then for stability. Theoretical work also suggests that increasing the batch size has a similar effect to decaying the learning rate (e.g. [Smith, Kindermans, & Le, 2018](https://arxiv.org/abs/1711.00489)), so you can think of this as an extra bit of learning rate annealing.
* I use the AdamW optimizer ([Loshchilov and Hunter, 2017](https://arxiv.org/abs/1711.05101)), with weight decay applied to all parameters except bias and LayerNorm weights. To save memory and speed up training, I use the 8-bit implementation from the [`bitsandbytes` library](https://github.com/TimDettmers/bitsandbytes). Gradients are clipped to a maximum norm of 0.5 for stability.
* I did not find the one-cycle learning rate schedule from the Cramming paper to work right out of the box. The simplicity of the one-cycle learning rate (see [Smith & Topin, 2017](https://arxiv.org/abs/1708.07120)) appealed to me, but the suggested maximum learning rate of 1e-3 caused my model to diverge during training (in fact, it never reaches that learning rate). I had to use a much lower maximum learning rate of 2e-4. Maybe the training data is noisier in my setting, because I keep all the accents and special characters and such. This may require a smaller learning rate. Rather than warming up for half of the token budget and annealing for the second half, I follow Iszak et al. and warm up for 10% of the token budget and anneal for the remaining 90%.
* Finally, I used PyTorch's automatic mixed precision utilities to train in (mostly) `fp16`, which saves memory, allowing me to go from a microbatch size of 128 to 256. It was tough to get all the pieces of this right (make sure you unscale the gradients before you clip them, folks!) but it was definitely worth it, and not as hard as it sounds.

## Finetuning on GLUE
I fine-tune on a subset of GLUE tasks that includes CoLA, SST-2, QQP, STS-B, MNLI, QNLI, and RTE.

| Model | CoLA | SST-2 | QQP | STS-B | MNLI-(m/mm) | QNLI | RTE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BERT-base |
| BERT-large |
| Best Crammed BERT |
| My Crammed BERT |
So far, I've achieved a MLM loss of around 1.9! I plan to fine-tune and evaluate the model on a few downstream tasks to gauge how well it performs there. I'll update this section as I make progress.

## References
Links are in the text above. I'll put a full bibliography here when I get a chance.

# Acknowledgments
Special thanks to Jonas Geiping, an author of the Cramming paper who was exceptionally helpful and kind in answering my questions about the paper, code, and training details. I also owe thanks to Andrej Karpathy and Phil Wang (`lucidrains` on GitHub), whose clear Transformer implementations have been a huge help in understanding how to build one myself. I learned and borrowed so many tricks from reading their code—you should do the same!
