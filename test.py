import wandb
import torch
from torch.utils.data import DataLoader
import numpy as np
from layers import *
from data import *
from model import BERT, BERTConfig

def test_attention(batch_size, seq_len, d_model, d_qkv, n_heads):
    attn = Attention(d_model, d_qkv, n_heads)
    ein_attn = EinsumAttention(d_model, d_qkv, n_heads)
    ein_attn.load_state_dict(attn.state_dict())
    in_tensor = torch.randn((batch_size, seq_len, d_model))
    out = attn(in_tensor)
    ein_out = ein_attn(in_tensor)
    assert torch.equal(out, ein_out), "Got different results with vanilla implementation vs. einsum implementation."
    assert in_tensor.shape == out.shape, "Input and output are not the same shape."
    print("Attention test passed!")

def test_ffn(batch_size, seq_len, d_model, ffn_hidden_size):
    # With GEGLU
    ffn = FFN(True, d_model, ffn_hidden_size)
    in_tensor = torch.randn((batch_size, seq_len, d_model))
    out_tensor = ffn(in_tensor)
    assert in_tensor.shape == out_tensor.shape, "Input and output are not the same shape (GEGLU variant)."
    ffn2 = FFN(False, d_model, ffn_hidden_size)
    out_tensor2 = ffn2(in_tensor)
    assert in_tensor.shape == out_tensor2.shape, "Input and output are not the same shape (vanilla variant)."
    print("FFN test passed!")

def test_transformer_block(batch_size, seq_len, d_model, d_qkv, n_heads, ffn_hidden_size):
    # With GEGLU
    tb = TransformerBlock(d_model, d_qkv, n_heads, True, ffn_hidden_size)
    in_tensor = torch.randn((batch_size, seq_len, d_model))
    out_tensor = tb(in_tensor)
    assert in_tensor.shape == out_tensor.shape, "Input and output are not the same shape (GEGLU variant)."
    tb2 = TransformerBlock(d_model, d_qkv, n_heads, False, ffn_hidden_size)
    out_tensor2 = tb2(in_tensor)
    assert in_tensor.shape == out_tensor2.shape, "Input and output are not the same shape (vanilla variant)."
    print("Transformer block tests passed!")

def test_config():
    config = BERTConfig(
        vocab_size=32768,
        n_layers=12,
        max_seq_len=512,
        d_model=768,
        d_qkv=64,
        n_heads=12,
        ffn_geglu=True,
        ffn_hidden_size=2048,
        tie_weights=True
    )
    config2 = BERTConfig.from_yaml("configs/test_config.yaml")
    assert config == config2, "Config from yaml is not the same as the one created manually."
    print("Config test passed!")

def test_bert():
    config = BERTConfig(
        vocab_size=32768,
        n_layers=12,
        max_seq_len=512,
        d_model=768,
        d_qkv=64,
        n_heads=12,
        ffn_geglu=True,
        ffn_hidden_size=2048,
        tie_weights=True
    )
    model = BERT(config)
    model.get_decay_params()
    in_tensor = torch.randint(0, config.vocab_size, (10, config.max_seq_len))
    out_tensor = model(in_tensor)
    assert out_tensor.shape == torch.Size([10, config.max_seq_len, config.vocab_size]),\
        "Output should have shape (batch_size, seq_len, vocab_size)."
    print("BERT test passed!")

def test_filter_and_batch_encode():
    tokenizer = train_or_load_tokenizer(file_path="webtext/tokenizer.json")
    it = webtext_batch_iterator()
    documents = next(it)
    filtered = filter_and_batch_encode(documents, tokenizer)
    print("Original documents: ", len(documents))
    print("Filtered documents: ", len(filtered))

def test_bert_dataset():
    dataset = BERTDataset("webtext/webtext_train.bin", 32768, 128, 1)
    x, y, mask = next(iter(dataset))
    assert x.shape == torch.Size([128]), "Input should have shape (seq_len, )."
    assert y.shape == torch.Size([128]), "Mask should have shape (seq_len, )."
    assert mask.shape == torch.Size([128]), "Mask should have shape (seq_len, )."
    print("BERT dataset test passed!")

def test_bert_dataloader():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dataset = BERTDataset("webtext/webtext_train.bin", 32768, 128, 1)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    first_batch = {}
    for x, y, mask in dataloader:
        first_batch["x"] = x
        first_batch["y"] = y
        first_batch["mask"] = mask
        break
    assert first_batch["x"].shape == torch.Size([64, 128]), "Input should have shape (batch_size, seq_len)."
    assert first_batch["y"].shape == torch.Size([64, 128]), "Target should have shape (batch_size, seq_len)."
    assert first_batch["mask"].shape == torch.Size([64, 128]), "Mask should have shape (batch_size, seq_len)."
    print("BERT dataloader test passed!")

def test_overfit(max_steps, max_lr=1e-3):
    wandb.init(
        project="cramming-test",
    
        # track hyperparameters and run metadata
        config={
            "max_lr": max_lr,
            "architecture": "BERT",
            "dataset": "webtext",
            "max_steps": max_steps,
            "batch_size": 16,
            "seq_len": 128,
            "max_seqs": 128
        }
    )
    print("Overfitting 128 sequences (static masks) in 1000 steps...")
    dataset = BERTDataset("webtext/webtext_val.bin", 32768, 128, 1, max_seqs=128)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    config = BERTConfig(
        vocab_size=32768,
        n_layers=6,
        max_seq_len=128,
        d_model=768,
        d_qkv=64,
        n_heads=12,
        ffn_geglu=True,
        ffn_hidden_size=2048,
        tie_weights=True
    )
    model = BERT(config)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=max_steps, pct_start=0.33, 
                                div_factor=10000, final_div_factor=25000, anneal_strategy="linear")
    step = 0
    static_batches = []
    for x, y, mask in dataloader:
        step += 1
        optimizer.zero_grad()
        loss = model(x, targets=y, mask=mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()
        print(f"Step {step} | Loss: {round(loss.item(), 3)}")
        wandb.log({"loss": loss.item()})
        static_batches.append((x, y, mask))
        if step >= 8:
            break
    print("Re-using first 8 batches to overfit, avoiding dynamic masking.")
    while step < max_steps:
        step += 1
        optimizer.zero_grad()
        x, y, mask = static_batches[step % 8]
        loss = model(x, targets=y, mask=mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()
        print(f"Step {step} | Loss: {round(loss.item(), 3)}")
        wandb.log({"loss": loss.item()})

if __name__ == "__main__":
    # Test all the layers
    test_attention(4, 32, 512, 64, 8)
    test_ffn(4, 32, 512, 2048)
    test_transformer_block(4, 32, 512, 64, 8, 2048)
    test_config()
    test_bert()

    # Test the tokenization
    test_filter_and_batch_encode()

    # Make sure the dataset and dataloader work with webtext
    test_bert_dataset()
    test_bert_dataloader()

    # Overfit on a small dataset
    # test_overfit(max_steps=400, max_lr=1e-2)

    print("All tests passed!")