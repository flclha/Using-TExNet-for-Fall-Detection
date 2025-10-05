import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import copy
from typing import Optional, Any, Union, Callable
import copy
from typing import Optional, Any, Union, Callable

# 滑动窗口函数
def self_sliding(data, window_width, stride):  # window_width 每个窗口包含历史数据点的个数
        num_windows = (data.shape[0] - window_width) // stride + 1  # 窗口数量
        windows = []

        for i in range(num_windows):
            window = data[i * stride: i * stride + window_width]
            windows.append(window)

        return np.array(windows)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=11):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)

        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp().unsqueeze(0)

        pe[:, 0::2] = torch.sin(position * div_term)
        d_model-=1
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp().unsqueeze(0)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

# 学习率调度器，学习率根据训练轮数动态变换，达到更好地收敛效果

def lr_scheduler(epoch, lr, warmup_epochs, decay_epochs, initial_lr, base_lr, min_lr):
    if epoch <= warmup_epochs:
        pct = epoch / warmup_epochs
        return ((base_lr - initial_lr) * pct) + initial_lr
    if epoch > warmup_epochs and epoch < warmup_epochs+decay_epochs:
        pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
        return ((base_lr - min_lr) * pct) + min_lr

    return min_lr

# GLU层定义
class GLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLU, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.mul(x, torch.sigmoid(self.linear(x)))

class LowRankLayer(nn.Module): # 使用奇异值分解低秩逼近
    def __init__(self, input_dim, output_dim, rank):
        super(LowRankLayer, self).__init__()
        self.weight = nn.Parameter(torch.rand(input_dim, output_dim))  # 维度input_dim*output_dim
        self.rank = rank

    def forward(self, x):
        u, s, v = torch.svd(self.weight)  # 对weight进行奇异值分解
        s = s[:self.rank]  # s的维度为rank*rank
        self.weight.data = torch.mm(u[:, :self.rank], torch.mm(torch.diag(s), v[:, :self.rank]).t())  # u维度为input_dim*rank v维度为rank*output
        return F.linear(x, self.weight)


class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=2, stride=1, padding=1, dilation=1)  # (1,2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=2, stride=1, padding=1, dilation=3)  # (32,2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * (input_dim // 6), 128)  # 这一层的输入要和下一层的输出相匹配（第一个值算下来应该等于128）
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # 将输入张量的形状从 (batch_size, input_dim) 变为 (batch_size, 1, input_dim)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # 将特征图展平为向量
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, nhead):  # 对象初始化操作
        super(Transformer, self).__init__()
        #self.embedding = PositionalEmbedding(d_model=input_dim) 时序任务中不需要加入位置编码
        #self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=nhead)
        self.encoder = nn.TransformerEncoder(  # 第一子层：多头注意力机制，第二子层：MLP 前馈神经网络（两个全连接层，中间激活函数）
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead),
            num_layers=2
        )
        self.norm = nn.BatchNorm1d(input_dim)
        self.classifier = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):  # 定义数据流前向传播过程
        x = self.encoder(x)
        x = self.classifier(x)
        x = self.dropout(x)  # 添加此dropout层略有提升，但提升不大（小于1%），是否要保留该层
        return x


def dynamic_weight_adjustment(weight, accuracy):
        # 根据准确率动态调整权重
        alpha = 0.1  # 调整步长
        target_accuracy = 1  # 期望准确率
        weight_adjusted = weight * (1 - alpha) + alpha * (accuracy - target_accuracy)
        return weight_adjusted

def compute_accuracy(predictions, ground_truth_labels):
        # 计算准确率的函数，根据损失计算
        _, predicted_labels = torch.max(predictions, 1)
        correct_predictions = (predicted_labels == ground_truth_labels).sum().item()
        total_samples = ground_truth_labels.size(0)
        accuracy = correct_predictions / total_samples
        return accuracy






