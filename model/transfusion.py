import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DBG = False

class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length = 100):
        super().__init__()

        self.device = device

        ''' Input Embedding '''
        # self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        # self.pos_embedding = nn.Embedding(max_length, hid_dim)

        ''' Multiple Encoder Layers '''
        # we use multiple encoder layers (e.g., 6 in the original Transformer paper)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, input_src, input_mask):

        batch_size = input_src.shape[0]
        src_len = input_src.shape[1]
        # emb_output = self.tok_embedding(input_src)
        # pos_tensor = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # 스케일 값 사용 확인 필요
        # output = self.dropout(emb_output * self.scale + self.pos_embedding(pos_tensor))
        output = input_src.to(self.device)
        for layer in self.layers:
          output = layer(output, input_mask)

        return output
    

class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        ''' Multi Head self-Attention '''
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)

        ''' Positional FeedForward Layer'''
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):

        # TODO: write your code
        # src : (batch_size, src_length, hidden_dim)
        # _src : (batch_size, src_length, hidden_dim)
        _src, attention = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(self.dropout(_src) + src)
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(self.dropout(_src) + src)


        return src
    
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fcQ = nn.Linear(hid_dim, hid_dim)
        self.fcK = nn.Linear(hid_dim, hid_dim)
        self.fcV = nn.Linear(hid_dim, hid_dim)
        self.fcOut = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.device = device


    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]
        src_len = query.shape[1]
        k_len = key.shape[1]
        v_len = value.shape[1]
        # (b, s_l, n_h, h_d)
        Q = self.fcQ(query).view(batch_size, src_len, self.n_heads, self.head_dim)
        K = self.fcK(key).view(batch_size, k_len, self.n_heads, self.head_dim)
        V = self.fcV(value).view(batch_size, v_len, self.n_heads, self.head_dim)

        # (b, n_h, s_l, h_d)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)

        scaled_atten = (Q / (self.head_dim ** 0.5)) @ K.transpose(2, 3) # (b, n_h, s_l, s_l)
        if DBG: print("mask size:",mask.shape)
        if DBG: print("sc_attn:",scaled_atten.shape)
        if mask is not None:
            scaled_atten = scaled_atten.masked_fill(mask == 0, -1e9)
        attention = self.dropout(nn.functional.softmax(scaled_atten, dim=-1))
        # attention = nn.functional.softmax(scaled_atten, dim=-1)
        # print("V:",V.shape)
        # print("attn:",attention.shape)
        output = torch.matmul(attention, V)
        x = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hid_dim)
        x = self.fcOut(x)
        # output = self.dropout()
        x = self.dropout(x)

        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.device = device
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)


    def forward(self, x):

        x = self.fc1(x)
        x = self.dropout(torch.relu(x))
        x = self.fc2(x)


        return x
    

class TransFusion(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.fc_out = nn.Linear(512*14, 512)
        self.device = device


    def forward(self, src, trg):

        # batch_size, trg_len = trg.size()
        # subsequent_mask = (1 - torch.triu(torch.ones((1, trg_len, trg_len), device=self.device), diagonal=1)).bool()
        src = torch.cat((src, trg), dim=1)
        # print("src:",src.shape)
        src_mask = (src != self.src_pad_idx).unsqueeze(-2)
        src_mask = src_mask.any(dim=-1)
        # print("s_mask:",src_mask.shape)
        # trg_mask = (trg != self.trg_pad_idx).unsqueeze(-2) & subsequent_mask
        if DBG: print("src:",src.shape)
        if DBG: print("s_mask:",src_mask.shape)
        # if DBG: print("t_mask:",trg_mask.shape)
        

        enc_out = self.encoder(src, src_mask)
        # output, attention = self.decoder(trg, trg_mask, enc_out, src_mask)
        out = torch.flatten(enc_out, 1)
        out = self.fc_out(out)
        return out
        # return output, attention
    

# INPUT_DIM = len(SRC.vocab)
# OUTPUT_DIM = len(TRG.vocab)
#------------------------------------------parameters-----------------------------------------------------
if __name__ == '__main__':
    INPUT_DIM = None
    HID_DIM = 512
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1


    enc = Encoder(INPUT_DIM,
                HID_DIM,
                ENC_LAYERS,
                ENC_HEADS,
                ENC_PF_DIM,
                ENC_DROPOUT,
                device)

    dec=None

    SRC_PAD_IDX = -1
    TRG_PAD_IDX = None
    model_TransFusion = TransFusion(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    criterion = torch.nn.MSELoss()
    batch_size = 32  
    tensor1 = torch.zeros(batch_size, 7, 512)
    tensor2 = torch.zeros(batch_size, 7, 512)
    out = model_TransFusion(tensor1, tensor2)
    print(out.shape)
# 이렇게 하면 됨
#context_vector = model_TransFusion(batch['images'], batch['texts'])