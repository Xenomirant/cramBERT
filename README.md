# cramBERT
This repository contains my implementation of the BERT architecture and training, including the data pipeline to train on the OpenWebText2 dataset and fine-tune on a set of downstream tasks from the GLUE benchmark. I was inspired to carry out this project by recent work on efficient, low-budget BERT training. I came across the "Cramming paper" ([Cramming: Training a Language Model on a Single GPU in One Day](https://arxiv.org/abs/2212.14034)), and was impressed how much performance they could get out of a BERT that they trained for just 24 hours on low-end hardware. They also managed to test lots of trendy architecture and training modifications along the way, and so their paper is the perfect guide to training a BERT with all the bells and whistles. I took this paper as my starting point to train a BERT that I can call my very own! In my case, I'm training on a single A100 in Google Colab, which is a bit nicer and faster than what the authors had access to, but it supports a similar sort of minimalist, scarcity-mindset training setup.

## Data and Preprocessing
I train and validate the model using the OpenWebText2 dataset, an open-source reproduction of OpenAI's WebText dataset by EleutherAI, which is a subset of the larger Pile dataset. The Cramming authors experiment with several corpuses, including BookCorpus-Wikipedia, C4, and The Pile—I decided to stick with just one solid dataset. OpenWebText2 comprises over 17 million scraped documents, totaling around 65GB of uncompressed text. The dataset and code are available [here](https://github.com/EleutherAI/openwebtext2), along with preprocessing utilities. I directly borrowed some of this code to load the dataset from JSONL archives. It is licensed under the [MIT License](https://github.com/EleutherAI/openwebtext2/blob/master/LICENSE).

OpenWebText2 is already de-duplicated, so the only additional preprocessing steps I perform are filtering out documents that aren't in English, and filtering out documents that don't compress well when tokenized, as suggested by the Cramming paper. I experimented with some of the filtering techniques suggested by the creaters of the DeepMind Gopher model ([Rae et al., 2022](https://arxiv.org/abs/2112.11446)), but I found that they didn't remove many documents. I suspect that the full Common Crawl scrapes in Gopher's MassiveWeb dataset contain more junk than OpenWebText2, which curates links from Reddit submissions. 

Unlike the authors of the Cramming paper, I do not convert the text to lowercase or strip out special characters and accents before tokenization. Every document from the OpenWebText2 dataset is left as-is.

## Tokenization
I further depart from the Cramming paper in my decision to use a byte-level BPE tokenizer, as I want the model to be able to represent any text on the web, including the special characters 🥺 and accents I chose not to strip out. I use the HuggingFace Tokenizers library to train a BPE tokenizer from scratch on OpenWebText2, filtered to only English webpages. In the pre-tokenization step, the tokenizer applies NFC normalization, adds a prefix space to the start of each document (so that a word at the start of the document is considered the same as a word in the middle of a document), and splits on whitespace using the same regular expression as the GPT2 tokenizer. I use a vocabulary size of 32,768, the same size as the WordPiece vocabulary in the Cramming paper (and a [nice number that makes GPUs happy](https://twitter.com/karpathy/status/1621578354024677377)). After tokenization, all resulting tokens are packed into sequences of length 128, with documents separated by a special `[SEP]` token. This means no padding is required, so no computation is wasted on padded sequences, and packing uses every token, so no data is wasted with truncation.

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

## Results — MLM Loss
After training on close to 10 billion tokens (which is how much the Cramming paper managed to train on in 24 hours—it took me a bit less time than that), my BERT model reached a MLM loss of I've managed to train BERT with the MLM loss objective to a training loss of 1.74, and a validation loss of 1.75 (roughly equal because I don't re-use data). This is better than the ~1.85 MLM loss achieved in the Cramming paper, but shouldn't be directly compared due to different data, preprocessing, and tokenization. As you'll see, the downstream results on GLUE are worse overall, which I'll speculate about below.

## Results – Fine-tuning on GLUE
I fine-tune on a subset of GLUE tasks that includes CoLA, SST-2, QQP, STS-B, MNLI, QNLI, and RTE. Following the Cramming paper, I restrict myself to a global hyperparameter setting for all downstream tasks (rather than tuning for each task separately), and fine-tune for 5 epochs on each task. I use the same batch size of 16, initial learning rate of 1.0e-4, and cosine decay schedule. Dropout of 0.1 is used for fine-tuning. I report the GLUE-dev results from the best epoch (not necessarily the last, as overfitting may occur for some tasks).

In the table below, I compare these results to the original BERT results, RoBERTa, and some models from the Cramming paper. Note that the original BERT paper only fine-tuned for 3 epochs, and reported results on the GLUE test set, rather than the development set. In addition to the original BERT and RoBERTa results, I report results from the Cramming paper, including (a) a BERT-base with no pre-training fine-tuned for up to 5 epochs; (b) a fully-pretrained BERT-base checkpoint fine-tuned for up to 5 epochs; and (c) the best "crammed" BERT fine-tuned for up to 5 epochs.

| Model | CoLA | SST-2 | QQP | STS-B | MNLI-(m/mm) | QNLI | RTE | Average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original BERT-base (GLUE test) | 52.1 | 93.5 | 71.2 | 85.8 | 84.6/83.4 | 90.5 | 66.4 | 78.4 |
| Original BERT-large (GLUE test) | 60.5 | 94.9 | 72.1 | 86.5 | 86.7/85.9 | 92.7 | 70.1 | 81.2 |
| RoBERTa (GLUE test) | 67.8 | 96.7 | 90.2 | 92.2 | 90.8/90.2 | 98.9 | 88.2 | 89.4 |
| Cramming Paper, BERT-base, no pretrain (GLUE dev) | 0.0 | 79.9 | 68.6 | 17.8 | 34.1/34.1 | 50.0 | 47.3 | 41.5 |
| Cramming Paper, Fully Trained BERT-base (GLUE dev) | 56.5 | 91.9 | 87.7 | 86.7 | 83.2/83.4 | 90.6 | 59.2 | 79.9 |
| Cramming Paper, Crammed BERT-base (GLUE dev) | 44.5 | 92.2 | 87.3 | 84.6 | 83.9/84.1 | 89.5 | 53.8 | 77.5 |
| My Crammed BERT-base (GLUE dev) | 18.4 | 90.3 | 79.9 | 83.1 | 78.7/78.8 | 87.1 | 60.3 | 72.1 |

The performance of my crammed BERT is not too shabby–a little bit worse than the Cramming paper, but better on some tasks, and miles ahead of a BERT with no pre-training. I suspect that the worse performance can be attributed to my setup: I made pretraining more complicated by not lower-casing everything, stripping out accents, and so on. I also trained on only OpenWebText2, which is sort of out-of-distribution for GLUE. I'd expect that this leads to a model that a better-equipped generalist for understanding arbitrary Internet text, but less well-suited for NLP benchmarks than a model trained on simpler text from books, news articles, and Wikipedia.

## Reflections and Future Work
People say that training a BERT is hard, and they're not wrong! I learned a lot in this process. Getting the training loop exactly right is especially tricky. Once you start adding bells and whistles like gradient accumulation, gradient clipping, learning rate scheduling, grouping parameters for weight decay, logging, mixed precision, etc. (which are all pretty simple on their own!) it starts to get busy, and if you forget one thing, or put things in the wrong order, the loss doesn't go down and you don't know why. (In my case, I wasn’t unscaling gradients before clipping. Whoops!) This makes me grateful that we have nice training interfaces like HuggingFace and Lightning, so that the "educational" experience I had is optional. 😉

Another takeaway—I really really wish that there was more distilled, available knowledge about how to initialize Transformer layers in a sane way. There's a lot of interesting research in this area, like [T-Fixup](https://proceedings.mlr.press/v119/huang20f.html), but I'm less interested in long proofs and the ability to stabilize training without a LayerNorm, and more interested in something simple that just works with typical Transformers. I wish someone would write a blog post or literature review about it! Maybe I'll have to. Without clear guidelines, there's a lot of guess-and-check involved, which was educational for me, but not the most fun part of the project.

I also experienced firsthand that memory- and time-efficient data loading into models is a big engineering challenge. Naive PyTorch dataset/dataloader implementations cause problems with big datasets (they don't share memory efficiently)—even fancier things I tried to build with memmaps fell apart with WebText-scale data (which is tiny compared to what SOTA Transformers are trained on!) In the future, I'm definitely going to just start out with either sharding data (which is what I ended up doing), or streaming it. Other memory-saving tricks I thought were clever didn't work, and I didn't find that out until midway through a training run. 😞 Relatedly, it’s super important to find ways to test incremental changes that don’t involve training the model for hours and hours only to have the edge case you were trying to fix crash everything. Avoid this by investing early in writing tests, systems to avoid and/or recover from loss spikes, resume training where you left off, etc. Sorry to all the polar bears whose ice caps I've been melting with all the wasted GPU-hours. 🥺

Finally, I was surprised by how much I had to "hack together" the pieces to evaluate on GLUE. This experience made me grateful for all the work that EleutherAI has put into making a plug-and-play [unified evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness) for causal LMs—when I train a GPT-style model, the downstream evaluation will be much more straightforward!

As for future work or things I'd do differently: I am curious how much performance would improve with better masking strategies. The naive MLM masking strategy is not optimal—Google has since released an update that masks whole words (see [the BERT repository](https://github.com/google-research/bert)), which makes the pre-training task more challenging (it's too easy when half of a word is masked and you just have to predict the other half). This comes out of the box if you use WordPiece tokenization, but I'd have to hack it together myself to work with byte-level BPE. Other work suggests that masking even larger *spans* (see [Joshi et al., 2019](https://arxiv.org/abs/1907.10529)) can lead to better performance. Also, if I was going to do it all again, and my ultimate goal was the best GLUE performance, I'd probably develop a more balanced corpus that includes more academic and literary text. I think the model did pretty well given that it only saw Internet text!

## References
[1] J. Geiping and T. Goldstein, “Cramming: Training a Language Model on a Single GPU in One Day,” arXiv:2212.14034 [cs.CL], Dec. 2022
[2] EleutherAI, “openwebtext2,” GitHub, 2022. [Online]. Available: https://github.com/EleutherAI/openwebtext2.
[3] J. Rae et al. "Scaling language models: Methods, analysis & insights from training gopher." arXiv:2112.11446 [cs.CL], 2021.
[4] J. Devlin, M. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” arXiv:1810.04805 [cs.CL], Oct. 2018.
[5] P. Izsak, M. Berchansky and O. Levy, “How to Train BERT with an Academic Budget,” arXiv:2104.07705 [cs.CL], Apr. 2021.
[6] S. L. Smith, P.-J. Kindermans, C. Ying, and Q. V. Le, “Don’t Decay the Learning Rate, Increase the Batch Size,” arXiv:1711.00489 [cs.LG], Nov. 2017.
[7] D. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” arXiv:1711.05101 [cs.LG], Nov. 2017.
[8] T. Dettmers, M. Lewis, S. Shleifer, and L. Zettlemoyer, “8-bit Optimizers via Block-wise Quantization,” arXiv:2110.02861 [cs.LG], Oct. 2021.
[9] L. N. Smith and N. Topin, “Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates,” arXiv:1708.07120 [cs.LG], Aug. 2017.
[10] X. S. Huang, et al. "Improving transformer optimization through better initialization." International Conference on Machine Learning. PMLR, 2020.
[11] M. Joshi, D. Chen, Y. Liu, D. S. Weld, L. Zettlemoyer, and O. Levy, “SpanBERT: Improving Pre-training by Representing and Predicting Spans,” arXiv:1907.10529 [cs], Jul. 2019

# Acknowledgments
Special thanks to Jonas Geiping, an author of the Cramming paper who was exceptionally helpful and kind in answering my questions about the paper, code, and training details. I also owe thanks to Andrej Karpathy and Phil Wang (`lucidrains` on GitHub), whose clear Transformer implementations have been a huge help in understanding how to build one myself. I learned so many tricks from reading their code–it's a great way to find some "ghost knowledge" that isn't always front-and-center in famous papers.
