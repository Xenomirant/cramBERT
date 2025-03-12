import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from collections import deque
import typing
import scipy
import numpy as np
from chinese_whispers import chinese_whispers, aggregate_clusters
import torch_scatter as t_sca
import einops
import networkx as nx
from numba import jit, njit
import pathlib
from IsoScore.IsoScore import IsoScore
import multiprocessing as mp

isoscore_dict = {"x_prelim":[], "x_new": []}

# Adapted from Andrej Karpathy's "nanoGPT"
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional weight & bias. PyTorch doesn't support simply bias=False,
    this module allows both, neither, or just one of [weight, bias]. """

    def __init__(self, d_model, weight=True, bias=False):
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model)) if weight else None
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, (self.d_model,), self.weight, self.bias, 1e-5)

class RMSNorm(nn.Module):
    def __init__(self, d_model, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d_model
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d_model))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d_model))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

    
class PreNormAndAdd(nn.Module):
    def __init__(self, d_model, sublayer, use_rms=False):
        super().__init__()
        if use_rms:
            self.norm = LayerNorm(d_model, bias=False)
        else:
            self.norm = RMSNorm(d_model, bias=False)
        self.sublayer = sublayer
    
    def forward(self, X, **kwargs):
        return X + self.sublayer(self.norm(X), **kwargs)


class SemanticPreNormAndAdd(nn.Module):
    def __init__(self, d_model, sublayer, use_rms=False):
        super().__init__()
        if use_rms:
            self.norm = LayerNorm(d_model, bias=False)
        else:
            self.norm = RMSNorm(d_model, bias=False)
        self.sublayer = sublayer    
        self.optimizer = torch.optim.SGD(self.parameters(), lr=3e-5, momentum=0.9, maximize=True) # try to rewrite as differentiable later

    def forward(self, X, **kwargs):
        # additional layerwise optimization step 
        if self.training:
            x_prelim = self.sublayer(self.norm(X), **kwargs)
            entropy = self.semantic_norm(x_prelim)
            # print(entropy)
            entropy.backward(
                retain_graph=True, 
                inputs=list(self.parameters())
            )
            # print([(x, x.grad) for x in self.parameters()])
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            
            x_new = self.sublayer(self.norm(X), **kwargs)

            self.compute_isoscore(x_prelim, x_new)
            
            return X + x_new
        return X + self.sublayer(self.norm(X), **kwargs)

    def semantic_norm(self, x):
        x = einops.rearrange(x, "b seq dim -> (b seq) dim")
        # construct mst (better to rewrite in numba)
        x_cpu = x.detach().cpu().numpy()

        index = self.graph_clustering(x_cpu)
        
        # select cluster centroids
        clust_cent = t_sca.scatter_mean(x, index=index, dim=0)
        cdist = torch.cdist(clust_cent, clust_cent)
        cdist = torch.triu(cdist)

        # compute dist entropy
        cdist = cdist[cdist!=0]
        cdist = cdist/cdist.sum()
        # 
        entropy = -(cdist * torch.log(cdist)).sum()

        return entropy

    @staticmethod
    # @jit(nopython=True)
    def graph_clustering(x_cpu):
        g = scipy.spatial.distance.cdist(x_cpu, x_cpu)
        g = nx.from_numpy_array(g)
        g = nx.minimum_spanning_tree(g)
        # perform graph clustering
        chinese_whispers(g, weighting="top", iterations=20)
        index = torch.zeros(x_cpu.shape[0], dtype=torch.int64, device="xpu")
            
        for i, cluster in enumerate(aggregate_clusters(g).values()):
            for j in cluster:
                index[j] = i

        return index

    @staticmethod
    def compute_isoscore(x_prelim, x_new):

        global isoscore_dict

        write_path = pathlib.Path("./logs")
        x_prelim = einops.rearrange(x_prelim, "b seq dim -> (b seq) dim")
        x_prelim = x_prelim.detach().cpu().numpy()
        x_new = einops.rearrange(x_new, "b seq dim -> (b seq) dim")
        x_new = x_new.detach().cpu().numpy()

        isoscore_dict["x_prelim"].append(x_prelim)
        isoscore_dict["x_new"].append(x_new)

        # as we have 12 layers in the model and 24 sublayers therefore
        if len(isoscore_dict) == 2500:
            for key in isoscore_dict:
                splitted_list = [isoscore_dict[key][i::25] for i in range(25)]
                for i, item in enumerate(splitted_list):
                    proc = mp.Process(target=_compute_isoscore,
                                      kwargs={"matrix": item,
                                             "write_to_file": write_path.joinpath(f"isoscore/{key}/layer_{i}.score")}, daemon=True)
                    proc.start()
                

class EmbeddingSemanticNorm(SemanticPreNormAndAdd):

    def __init__(self, vocab_size, d_model, max_seq_len, use_rms=False):
        super().__init__(d_model=d_model, sublayer=None)
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        if use_rms:
            self.emb_norm = LayerNorm(d_model, bias=False)
        else:
            self.emb_norm = RMSNorm(d_model, bias=False)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=3e-5, momentum=0.9, maximize=True) 
    
    def forward(self, X, **kwargs):
        # additional layerwise optimization step 
        token_embs = self.token_emb(X)
        pos_embs = self.pos_emb[:, :X.shape[1], :]
        x_prelim = self.token_emb(X) + self.pos_emb[:, :X.shape[1], :]

        if self.training:
            entropy = self.semantic_norm(x_prelim)
            # print(entropy)
            entropy.backward(
                retain_graph=True, 
                inputs=list(self.parameters())
            )
            # print([(x, x.grad) for x in self.parameters()])
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            x_new = self.token_emb(X) + self.pos_emb[:, :X.shape[1], :]
            x_new = self.emb_norm(x_new)

            self.compute_isoscore(x_prelim, x_new)
            
            return x_new
        return x_prelim


def _compute_isoscore(matrix, write_to_file):
    matrix = np.array(matrix).reshape(-1, 768)
    isoscore = IsoScore(matrix)

    with open(write_to_file, "a") as f:
        print(isoscore, end='\n', file=f)
    return None
                              
        
class Attention(nn.Module):
    def __init__(self, d_model, d_qkv, n_heads, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_qkv = d_qkv
        self.scale = d_qkv**0.5
        self.n_heads = n_heads
        self.to_qkv = nn.Linear(d_model, 3 * d_qkv * n_heads, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_qkv * n_heads, d_model, bias=False)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, X, mask=None):
        b, s, _ = X.shape
        q, k, v = self.to_qkv(X).view(b, s, self.n_heads, self.d_qkv, 3).unbind(dim=-1)
        attn_scores = q.transpose(1, 2) @ k.permute((0, 2, 3, 1)) / self.scale
        if mask is not None:
            # print("mask is not None")
            # Fill padding with -inf
            # Mask is shape (b, s) and attn_scores is shape (b, n_heads, s, s)
            # We need to unsqueeze mask to shape (b, 1, 1, s) and fill where mask is 0 
           # (padding) along the key dimension
            mask = mask.unsqueeze(1).unsqueeze(1)
            attn_scores.masked_fill_(~mask, float('-inf'))
        attn_weights = self.attn_dropout(F.softmax(attn_scores, dim=-1))
        # print("weights:", attn_weights[0, 0, :, :])
        # v_transp = v.transpose(1, 2)
        # print("v_transp (s * d_qkv):", v_transp[0, 0, :, :])
        attn_out = attn_weights @ v.transpose(1, 2)
        
        # print(attn_out[0, 0, :, :]) # b, nh, s, d_qkv
        return self.out_dropout(self.out_proj(attn_out.transpose(1, 2).flatten(-2)))

class FFN(nn.Module):
    def __init__(self, geglu, d_model, hidden_size, dropout=0.0):
        super().__init__()
        self.geglu = geglu
        if geglu:
            self.fc1 = nn.Linear(d_model, 2 * hidden_size, bias=False)
        else: 
            self.fc1 = nn.Linear(d_model, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, d_model, bias=False)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, X):
        if self.geglu:
            a, b = self.fc1(X).chunk(2, dim=-1)
            return self.out_dropout(self.fc2(a * F.gelu(b, approximate='tanh')))
        else:
            return self.out_dropout(self.fc2(F.gelu(self.fc1(X), approximate='tanh')))

norm_layer_map = {
    "PreLayerNorm": PreNormAndAdd,
    "SemanticPreLayerNorm": SemanticPreNormAndAdd,
}

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_qkv, n_heads, ffn_geglu, ffn_hidden_size, dropout=0.0, use_rms=False, norm_layer="PreLayerNorm"):
        super().__init__()
        NormLayer = norm_layer_map[norm_layer]
        self.attn = NormLayer(d_model, Attention(d_model, d_qkv, n_heads, dropout=dropout), use_rms=use_rms)
        self.ffn =  NormLayer(d_model, FFN(ffn_geglu, d_model, ffn_hidden_size, dropout=dropout), use_rms=use_rms)

    def forward(self, X, mask=None):
        return self.ffn(self.attn(X, mask=mask))