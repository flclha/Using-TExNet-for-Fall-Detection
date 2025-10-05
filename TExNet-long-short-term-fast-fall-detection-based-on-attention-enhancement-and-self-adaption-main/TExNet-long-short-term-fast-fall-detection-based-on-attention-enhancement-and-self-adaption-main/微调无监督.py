import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from functions2 import lr_scheduler, CNN, Transformer, self_sliding
import pandas as pd
import os
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# 定义设备
device = 'cpu'
num_epochs_finetune = 100
window_size = 350
# 用于训练的第二个数据集数据
new_data = pd.read_excel('自建数据-平衡后的数据.xlsx')  # 引入新的数据文件
new_X = new_data.iloc[:, 1:7].values
y = new_data.iloc[:, 13].values

#需要先用滑动窗口，将每350个划分为一组
#划分滑动窗口
X_train_sliding = self_sliding(new_X, window_width=window_size, stride=70)  # 一次试验200个数据
y_train_sliding = self_sliding(y, window_width=window_size, stride=70)  # 一次试验200个数据

num_samples_train = X_train_sliding.shape[0]  # 训练集样本数量
num_windows_train = X_train_sliding.shape[1]  # 训练集窗口数量
X_train_reshaped = np.reshape(X_train_sliding, (num_samples_train * num_windows_train, 6))  # 需要和输入特征个数相同
y_train_reshaped = np.reshape(y_train_sliding, (num_samples_train * num_windows_train,))

new_X = X_train_reshaped
y = y_train_reshaped

# 进行微调
batch_size = 128
scaler = StandardScaler()  # 归一化处理
scaler.fit(new_X)  # 先进行fit操作
new_X = scaler.transform(new_X)

train_data = torch.utils.data.TensorDataset(torch.Tensor(new_X))
new_train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)  # 定义批大小 考虑是否要打乱,必须和预训练相同


class MyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyNet, self).__init__()
        self.cnn = CNN(input_dim, output_dim)
        self.transformer = Transformer(input_dim, output_dim, nhead=6)
        self.weight_cnn = nn.Parameter(torch.ones(1))
        self.weight_transformer = nn.Parameter(torch.ones(1))

        self.classifier = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        cnn_output = self.cnn(x)
        transformer_output = self.transformer(x)

        # 按权重分配拼接结果
        weighted_output = self.weight_cnn * cnn_output + self.weight_transformer * transformer_output
        output = self.classifier(weighted_output)
        return output


model2 = MyNet(input_dim=6, output_dim=2).to(device)  # input_dim 输入维度为X列数，输出维度为分类数
# 在第二个模型中加载大模型的参数
model2.load_state_dict(torch.load('large_modellv.pth'))
optimizer_low_rank = optim.SGD([
    {'params': model2.cnn.parameters(), 'lr': 0.001},
    {'params': model2.transformer.parameters(), 'lr': 0.001},
    {'params': model2.classifier.parameters(), 'lr': 0.01}
], momentum=0.9)
# weight_decay添加正则化，相当于L2正则化（可改参数）

def unsupervised_loss(outputs):
    # 定义无监督损失函数，例如均方误差损失
    return torch.mean(torch.pow(outputs, 2))

train_acc_list = []
train_loss_list = []
test_acc_list = []
test_loss_list = []
train_acc = 0
test_acc = 0

# 增加学习率调度器，参数需改
warmup_epochs = 5  # 预热阶段轮数
decay_epochs = 40  # 衰减阶段轮数
initial_lr = 1e-4  # 预热阶段学习率
base_lr = 1e-3  # 衰减阶段学习率
min_lr = 5e-5

for epoch in range(num_epochs_finetune):
    lr = lr_scheduler(epoch, optimizer_low_rank.param_groups[0]['lr'], warmup_epochs, decay_epochs, initial_lr, base_lr,
                      min_lr)
    optimizer_low_rank.param_groups[0]['lr'] = lr
    for i, (inputs,) in enumerate(new_train_loader):
        inputs = inputs.to(device)
        optimizer_low_rank.zero_grad()
        outputs = model2(inputs)
        loss = unsupervised_loss(outputs.squeeze(-1))  # 使用无监督损失函数计算损失值
        loss.backward()
        optimizer_low_rank.step()

    model2.eval()
    with torch.no_grad():
        predictions = []
        truths = []
        # 初始化计数器
        count1 = 0

        for inputs in new_train_loader:
            inputs = inputs[0].to(device)

            output = model2(inputs)
            _, predicted = torch.max(output, 1)
            predictions += predicted.tolist()
            truths += inputs

        new_predictions = []
        group_count = 0
        group_label = 0
        for i in range(len(predictions)):
            group_count += 1
            if predictions[i] == 1:
                group_label += 1
            if group_count == window_size:
                if group_label > 100:
                    new_predictions += [1] * window_size
                else:
                    new_predictions += [0] * window_size
                group_label = 0
                group_count = 0
                if new_predictions[i] == y[i]:
                    count1 += 1
        print(len(predictions))
        print(count1)
        accuracy1 = 1-abs(count1-690) /690  # 滑窗200 48422正例  242组 滑窗 350 350 正例 48422 138组  滑窗350 70 690组

    print(f'Epoch {epoch + 1}/{num_epochs_finetune}, Accuracy1: {accuracy1:.4f}')