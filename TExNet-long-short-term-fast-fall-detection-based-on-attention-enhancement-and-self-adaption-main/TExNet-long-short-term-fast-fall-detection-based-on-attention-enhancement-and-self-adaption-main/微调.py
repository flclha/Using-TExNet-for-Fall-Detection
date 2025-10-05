import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import torch.optim as optim
from IRM import irm_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from functions2 import lr_scheduler, CNN, Transformer, dynamic_weight_adjustment, compute_accuracy
from sklearn.metrics import roc_auc_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# 定义设备
device = 'cpu'
num_epochs_finetune = 1000
# 用于训练的第二个数据集数据
new_data = pd.read_excel('滤波原始数据2——时间分解.xlsx')  # 引入新的数据文件
new_X = new_data.iloc[:, 1:13].values
new_y = new_data.iloc[:, 13].values
new_env = new_data.iloc[:, 14].values

class MyNet(nn.Module):
    def __init__(self, input_dim, output_dim, rank):
        super(MyNet, self).__init__()
        self.cnn = CNN(input_dim, output_dim)
        self.transformer = Transformer(input_dim, output_dim, nhead=12)
        self.weight_cnn = nn.Parameter(torch.ones(1))
        self.weight_transformer = nn.Parameter(torch.ones(1))

        self.classifier = nn.Linear(output_dim, output_dim)

    def forward(self, x, ground_truth_labels):
        cnn_output = self.cnn(x)
        transformer_output = self.transformer(x)

        # 根据损失计算准确率动态调整权重
        cnn_accuracy = compute_accuracy(cnn_output, ground_truth_labels)
        transformer_accuracy = compute_accuracy(transformer_output, ground_truth_labels)

        self.weight_cnn.data = dynamic_weight_adjustment(self.weight_cnn, cnn_accuracy)
        self.weight_transformer.data = dynamic_weight_adjustment(self.weight_transformer, transformer_accuracy)

        # 按权重分配拼接结果
        weighted_output = self.weight_cnn * cnn_output + self.weight_transformer * transformer_output
        output = self.classifier(weighted_output)
        return output

model2 = MyNet(input_dim=12, output_dim=2, rank=4).to(device)  # input_dim 输入维度为X列数，输出维度为分类数
# 在第二个模型中加载大模型的参数
model2.load_state_dict(torch.load('large_modellv.pth'))
optimizer_low_rank = optim.SGD([
    {'params': model2.cnn.parameters(), 'lr': 0.001},
    {'params': model2.transformer.parameters(), 'lr': 0.001},
    {'params': model2.classifier.parameters(), 'lr': 0.01}
], momentum=0.9)
# weight_decay添加正则化，相当于L2正则化（可改参数）
criterion = nn.CrossEntropyLoss()  # 定义二元交叉熵损失函数，用于比较模型预测值和真实值的差异

# 计算准确率
def accuracy(predictions, truths):
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == truths[i]:
            correct += 1
    return correct / len(predictions)

train_acc_list = []
train_loss_list = []
test_acc_list = []
test_loss_list = []
val_acc_list = []
val_loss_list = []
train_acc = 0
test_acc = 0
val_acc = 0

# 增加学习率调度器，参数需改
warmup_epochs = 5  # 预热阶段轮数
decay_epochs = 20  # 衰减阶段轮数
initial_lr = 1e-4  # 预热阶段学习率
base_lr = 1e-3  # 衰减阶段学习率
min_lr = 5e-5

# 进行微调
model2.train()
batch_size = 128
scaler = StandardScaler()  # 归一化处理
scaler.fit(new_X)  # 先进行fit操作
new_X = scaler.transform(new_X)

# 第一步：将数据集划分为训练集和临时集（临时集包含测试集和验证集）
new_X_train, new_X_temp, new_y_train, new_y_temp, new_env_train, new_env_temp = train_test_split(
    new_X, new_y, new_env, test_size=0.3, random_state=42)

# 第二步：将临时集划分为测试集和验证集
new_X_test, new_X_val, new_y_test, new_y_val, new_env_test, new_env_val = train_test_split(
    new_X_temp, new_y_temp, new_env_temp, test_size=0.2 / 0.3, random_state=42)

train_data = torch.utils.data.TensorDataset(torch.Tensor(new_X_train), torch.Tensor(new_y_train), torch.Tensor(new_env_train))
test_data = torch.utils.data.TensorDataset(torch.Tensor(new_X_test), torch.Tensor(new_y_test), torch.Tensor(new_env_test))
val_data = torch.utils.data.TensorDataset(torch.Tensor(new_X_val), torch.Tensor(new_y_val), torch.Tensor(new_env_val))

new_train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)  # 定义批大小
new_test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
new_val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs_finetune):
    lr = lr_scheduler(epoch, optimizer_low_rank.param_groups[0]['lr'], warmup_epochs, decay_epochs, initial_lr, base_lr, min_lr)
    optimizer_low_rank.param_groups[0]['lr'] = lr
    for i, (inputs, labels, groups) in enumerate(new_train_loader):
        inputs, labels, groups = inputs.to(device), labels.to(device), groups.to(device)
        optimizer_low_rank.zero_grad()
        outputs = model2(inputs, labels)
        loss = criterion(outputs.squeeze(-1), labels.long())  # 使用irm_loss计算模型损失值
        loss.backward()
        optimizer_low_rank.step()

    model2.eval()
    with torch.no_grad():
        # 训练集
        predictions = []
        truths = []
        for inputs, labels, _ in new_train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            output = model2(inputs, labels)
            _, predicted = torch.max(output, 1)
            predictions += predicted.tolist()
            truths += labels.tolist()

        tn, fp, fn, tp = confusion_matrix(truths, predictions).ravel()

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        # 每一轮训练中计算准确率和损值
        train_acc = accuracy(predictions, truths)
        train_loss = loss.item()

        # 保存准确率和损失值
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        # 测试集
        predictions = []
        truths = []
        for inputs, labels, _ in new_test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            output = model2(inputs, labels)
            _, predicted = torch.max(output, 1)
            predictions += predicted.tolist()
            truths += labels.tolist()

        tn, fp, fn, tp = confusion_matrix(truths, predictions).ravel()

        test_sensitivity = tp / (tp + fn)
        test_specificity = tn / (tn + fp)

        # 计算测试集准确率
        test_acc = accuracy(predictions, truths)
        # 保存测试集准确率和损失值
        test_acc_list.append(test_acc)
        test_loss_list.append(loss.item())

        # 验证集
        predictions = []
        truths = []
        for inputs, labels, _ in new_val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            output = model2(inputs, labels)
            _, predicted = torch.max(output, 1)
            predictions += predicted.tolist()
            truths += labels.tolist()

        tn, fp, fn, tp = confusion_matrix(truths, predictions).ravel()

        val_sensitivity = tp / (tp + fn)
        val_specificity = tn / (tn + fp)

        # 计算验证集准确率
        val_acc = accuracy(predictions, truths)
        # 保存验证集准确率和损失值
        val_acc_list.append(val_acc)
        val_loss_list.append(loss.item())

        print(f'Epoch {epoch + 1} / {num_epochs_finetune}, Testing: Sensitivity: {test_sensitivity}, Specificity: {test_specificity}, Validation: Sensitivity: {val_sensitivity}, Specificity: {val_specificity}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Val Acc: {val_acc:.4f}')

# 绘制折线图
plt.figure(figsize=(10, 5))  # 设置图形窗口大小

# 绘制训练集、测试集和验证集准确率折线
plt.plot(range(1, num_epochs_finetune + 1), train_acc_list, label='Train Accuracy')
plt.plot(range(1, num_epochs_finetune + 1), test_acc_list, label='Test Accuracy')
plt.plot(range(1, num_epochs_finetune + 1), val_acc_list, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training, Testing and Validation Accuracy')
plt.legend()

plt.figure(figsize=(10, 5))  # 设置图形窗口大小
# 绘制训练集、测试集和验证集损失折线
plt.plot(range(1, num_epochs_finetune + 1), train_loss_list, label='Train Loss')
plt.plot(range(1, num_epochs_finetune + 1), test_loss_list, label='Test Loss')
plt.plot(range(1, num_epochs_finetune + 1), val_loss_list, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training, Testing and Validation Loss')
plt.legend()

# 显示图形
plt.show()