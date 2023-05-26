import torch.nn as nn
import math
from wrap_data import *


PAD_token = 0
BOS_token = 1
EOS_token = 2
UNK_token = 3


def build_model(input_lang, output_lang, max_seq_len, configs: dict):
    model = Transformer(enc_voc_size=len(input_lang.index2word),
                        dec_voc_size=len(output_lang.index2word),
                        generator=Generator(configs['model_dim'], len(output_lang.index2word)),
                        model_dim=configs['model_dim'],
                        n_head=configs['n_head'],
                        max_len=max_seq_len,
                        hidden_dim=configs['hidden_dim'],
                        n_layer=configs['n_layer'],
                        drop_prob=configs['drop_prob'],
                        src_pad_idx=configs['pad_token'],
                        trg_pad_idx=configs['pad_token'],
                        src_bos_idx=configs['bos_token'],
                        trg_bos_idx=configs['bos_token'],
                        src_eos_idx=configs['eos_token'],
                        trg_eos_idx=configs['eos_token'],
                        device=eval(configs['device']))
    return model


class PositionLayer(nn.Module):
    def __init__(self, max_len, embed_dim, device):
        super(PositionLayer, self).__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.encoding = torch.zeros(max_len, embed_dim, requires_grad=False, device=device)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * -(math.log(10000.0) /embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)].requires_grad_(False)


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, dropout=None, mask=None, e=1e9):  # fixed the eps
        batch_size, head, length, tensor_dim = k.size()
        k_t = k.transpose(-2, -1)
        score = (q @ k_t) / math.sqrt(tensor_dim)

        if mask is not None:
            score = score.masked_fill(mask == 0, -e)  # keep the shape of mask since the broadcast

        score = self.softmax(score)
        if dropout is not None:
            score = dropout(score)  # dropout attention

        v = score @ v
        return v, score


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, drop_prob, device):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_dim = self.embed_dim // self.n_head

        self.linear_q = nn.Linear(self.embed_dim, self.head_dim * self.n_head, device=device)
        self.linear_k = nn.Linear(self.embed_dim, self.head_dim * self.n_head, device=device)
        self.linear_v = nn.Linear(self.embed_dim, self.head_dim * self.n_head, device=device)
        self.scaled_dot_product_attention = ScaleDotProductAttention()

        self.linear_attention = nn.Linear(self.head_dim * self.n_head, self.embed_dim, device=device)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, q, k, v, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # add a dimension into mask
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)
        batch_size = k.size()[0]

        Q = q.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        K = k.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        V = v.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        context, _ = self.scaled_dot_product_attention(Q, K, V, self.dropout, mask)

        output = context.transpose(1, 2).contiguous().view(batch_size, -1, self.head_dim * self.n_head)
        del q
        del k
        del v
        del Q
        del K
        del V
        output = self.linear_attention(output)
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, model_dim, hidden_dim, drop_prob, device):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(model_dim, hidden_dim, device=device)
        self.linear2 = nn.Linear(hidden_dim, model_dim, device=device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, model_dim, device, eps=1e-6):  # fixed the eps
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(model_dim, device=device))
        self.beta = nn.Parameter(torch.zeros(model_dim, device=device))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.model_dim = model_dim
        self.tok_embedding = nn.Embedding(vocab_size, model_dim, device=device)
        self.pos_embedding = PositionLayer(max_len, model_dim, device=device)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        tok_embedding = self.tok_embedding(x) * math.sqrt(self.model_dim)  # multiply square root of model dim
        pos_embedding = self.pos_embedding(x)
        return self.dropout(tok_embedding + pos_embedding)


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, hidden_dim, n_head, drop_prob, device):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, n_head, drop_prob, device=device)
        self.layer_norm1 = LayerNorm(model_dim, device=device)
        self.dropout1 = nn.Dropout(drop_prob)
        self.feed_forward = PositionWiseFeedForward(model_dim, hidden_dim, drop_prob, device=device)
        self.layer_norm2 = LayerNorm(model_dim, device=device)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, s_mask):
        # the norm order seems to matter
        xnorm = self.layer_norm1(x)
        x = x + self.dropout1(self.attention(xnorm, xnorm, xnorm, mask=s_mask))
        xnorm = self.layer_norm2(x)
        x = x + self.dropout2(self.feed_forward(xnorm))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, model_dim, hidden_dim, n_head, drop_prob, device):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(model_dim, n_head, drop_prob, device=device)
        self.layer_norm1 = LayerNorm(model_dim, device=device)
        self.dropout1 = nn.Dropout(drop_prob)
        self.enc_dec_attention = MultiHeadAttention(model_dim, n_head, drop_prob, device=device)
        self.layer_norm2 = LayerNorm(model_dim, device=device)
        self.dropout2 = nn.Dropout(drop_prob)
        self.feed_forward = PositionWiseFeedForward(model_dim, hidden_dim, drop_prob, device=device)
        self.layer_norm3 = LayerNorm(model_dim, device=device)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, dec, enc, s_mask, t_mask):
        # the norm order seems to matter
        decnorm = self.layer_norm1(dec)
        dec = dec + self.dropout1(self.self_attention(decnorm, decnorm, decnorm, mask=t_mask))
        decnorm = self.layer_norm2(dec)
        dec_lookup = dec + self.dropout2(self.enc_dec_attention(decnorm, enc, enc, mask=s_mask))
        declookup_norm = self.layer_norm3(dec_lookup)
        dec_lookup = dec_lookup + self.dropout3(self.feed_forward(declookup_norm))
        return dec_lookup


class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, model_dim, hidden_dim, n_head, n_layer, drop_prob, device):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(model_dim=model_dim,
                                              max_len=max_len,
                                              vocab_size=enc_voc_size,
                                              drop_prob=drop_prob,
                                              device=device
                                              )
        self.layers = nn.ModuleList([EncoderLayer(model_dim,
                                                  hidden_dim,
                                                  n_head,
                                                  drop_prob,
                                                  device=device) for _ in range(n_layer)])
        # add norm layer at the end
        self.norm = LayerNorm(model_dim, device=device)
    def forward(self, x, s_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, s_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, model_dim, hidden_dim, n_head, n_layer, drop_prob, device):
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(model_dim=model_dim,
                                              max_len=max_len,
                                              vocab_size=dec_voc_size,
                                              drop_prob=drop_prob,
                                              device=device)
        self.layers = nn.ModuleList([DecoderLayer(model_dim,
                                                  hidden_dim,
                                                  n_head,
                                                  drop_prob,
                                                  device=device) for _ in range(n_layer)])
        self.norm = LayerNorm(model_dim, device=device)

    def forward(self, trg, enc_src, src_mask, trg_mask):
        trg = self.embedding(trg)
        for layer in self.layers:
            trg = layer(trg, enc_src, src_mask, trg_mask)
        trg = self.norm(trg)  # add another norm layer before linear classification layer
        return trg


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)
        # return log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, enc_voc_size, dec_voc_size, generator, model_dim, n_head,
                 max_len, hidden_dim, n_layer, drop_prob, src_pad_idx, trg_pad_idx,
                 src_bos_idx, trg_bos_idx, src_eos_idx, trg_eos_idx, device):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.src_bos_idx = src_bos_idx
        self.trg_bos_idx = trg_bos_idx
        self.src_eos_idx = src_eos_idx
        self.trg_eos_idx = trg_eos_idx
        self.enc_voc_size = enc_voc_size
        self.dec_voc_size = dec_voc_size
        self.max_len = max_len
        self.encoder = Encoder(model_dim=model_dim,
                               n_head=n_head,
                               max_len=max_len,
                               hidden_dim=hidden_dim,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layer=n_layer,
                               device=device)
        self.decoder = Decoder(model_dim=model_dim,
                               n_head=n_head,
                               max_len=max_len,
                               hidden_dim=hidden_dim,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layer=n_layer,
                               device=device)
        self.generator = generator
        self.device = device

    @staticmethod
    def make_pad_mask(tensor, pad_idx, device=None):
        pad_mask = (tensor != pad_idx).unsqueeze(-2)
        if device:
            pad_mask = pad_mask.to(device)
        return pad_mask

    @staticmethod
    def make_no_peak_mask(tensor, device=None):
        # tensor.size(1)
        mask = torch.triu(torch.ones(1, tensor.size(1), tensor.size(1)), diagonal=1).type(torch.uint8)
        if device:
            mask = mask.to(device)
        return mask == 0

    def forward(self, src, trg):
        # there are many coding style to generate mask
        src_mask = self.make_pad_mask(src, self.src_pad_idx, self.device)
        trg_mask = self.make_pad_mask(trg, self.trg_pad_idx,
                                      self.device) & self.make_no_peak_mask(trg, self.device)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, src_mask, trg_mask)
        output = self.generator(output)
        return output


