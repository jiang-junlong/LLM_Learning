import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.categories.baseclasses import Class
from torch.ao.nn.quantized import MultiheadAttention


def attention(query, key, value, mask, dropout):
    d_k=query.size(-1)
    socore=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not  None:
        socore=socore.masked_fill(mask==0,-1e9)
    p_attn= F.softmax(socore,dim=-1)
    if dropout:
        p_attn=dropout(p_attn)
    return torch.matmul(p_attn,value),p_attn

def clones(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):
    def __init__(self,h ,d_model, dropout=0.1):
        super(MultiHeadAttention).__init__()
        assert d_model%h==0
        self.d_k=d_model//h         # 每个头的维度
        self.h=h                    # 注意力头的数量
        self.linears=clones(nn.Linear(d_model,d_model),4)
        self.attn=None
        self.dropout=nn.Dropout(p=dropout)
    def forward(self,query, key, value,mask=None):
        if mask is not None:
            mask=mask.unsqueeze(-1) #增加一个维度，使掩码形状从 (batch_size, seq_len) 变为 (batch_size, seq_len, 1)
        nbatches=query.size(0)
        Q, K, V = \
        [l(x).view(nbatches,-1,self.h,self.d_k).transpose(1,2) for l,x in zip(self.linears,(query,key,value))]
        # 对query、key、value 分别应用线性层（self.linears[0:3]）
        # 输入形状：(batch_size, seq_len, d_model)
        # 线性变换后：(batch_size, seq_len, d_model)
        # view重塑为：(batch_size, seq_len, h, d_k)
        # transpose(1, 2)交换维度，变为(batch_size, h, seq_len, d_k)
        x,self.attn=attention(Q,K,V,mask=mask,dropout=self.dropout)
        x=x.transpose(1,2).contiguous().view(nbatches,-1,self.h*self.d_k)
        return self.linears[-1](x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        '''
        :param d_model: 词嵌入维度
        :param dropout: dropout触发概率
        :param max_len: 每个句子的最大长度
        '''
        super(PositionalEncoding,self).__init__()
        self.dropout=nn.Dropout(p=dropout)



class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder,self).__init__()