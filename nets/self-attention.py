from math import sqrt

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.dim_k = dim_k
        self.dim_v = dim_v

        # 定义线性变换函数
        self.linear_q = nn.Linear(input_dim, dim_k, bias=False)
        self.linear_k = nn.Linear(input_dim, dim_k, bias=False)
        self.linear_v = nn.Linear(input_dim, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        # x: batch, n, input_dim
        # 根据文本获得相应的维度

        batch, n, input_dim = x.shape
        print(batch)
        print(n)
        print(input_dim)
        assert input_dim == self.input_dim

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v
        # q*k的转置 并*开根号后的dk
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        # 归一化获得attention的相关系数
        dist = torch.softmax(dist, dim=-1)  # batch, n, n
        # attention系数和v相乘，获得最终的得分
        att = torch.bmm(dist, v)
        return att


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature
                attention: B * N * N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B*N*C/8
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B*C*N/8
        energy = torch.bmm(proj_query, proj_key)  # batch的matmul B*N*N
        attention = self.softmax(energy)  # B * (N) * (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B * C * N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B*C*N
        out = out.view(m_batchsize, C, width, height)  # B*C*H*W

        out = self.gamma * out + x
        # out = self.gamma * out
        return out   #, attention
if __name__ == '__main__':
    x = torch.randn(4,8,18,18)
    print(x)
    print(x.shape)
    at = Self_Attn(8)
    res = at(x)[0]
    print(res.shape)
    print(res)
