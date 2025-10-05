import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import torch.optim as optim
from IRM import irm_loss
from sklearn.preprocessing import StandardScaler
from functions2 import self_sliding, lr_scheduler, CNN, Transformer, compute_accuracy, dynamic_weight_adjustment
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
#公开数据集只训练，微调：训练自建数据集并测试、验证
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = 'cpu'

# 加载数据
data = pd.read_excel('Upfall4完整版数据(标注01）.xlsx')  # 引入处理数据文件
#data = pd.read_excel('没滤波原始数据2——时间分解.xlsx')  # 引入处理数据文件
X = data.iloc[:, 1:13].values
y = data.iloc[:, 13].values
env_labels = data.iloc[:, 14].values  # 获取环境标签列

# 数据预处理
scaler = StandardScaler()  # 归一化处理
X = scaler.fit_transform(X)

  # 划分滑动窗口
X_train_sliding = self_sliding(X, window_width=200, stride=100)  # 一次试验200个数据
y_train_sliding = self_sliding(y, window_width=200, stride=100)
env_train_sliding = self_sliding(env_labels, window_width=200, stride=100)

num_samples_train = X_train_sliding.shape[0]  # 训练集样本数量#
num_windows_train = X_train_sliding.shape[1]  # 训练集窗口数量
#
X_train_reshaped = np.reshape(X_train_sliding, (num_samples_train * num_windows_train, 12))  # 需要和输入特征个数相同
y_train_reshaped = np.reshape(y_train_sliding, (num_samples_train * num_windows_train,))
env_train_reshaped = np.reshape(env_train_sliding, (num_samples_train * num_windows_train,))

X_combined = X_train_reshaped
y_combined = y_train_reshaped

# 规定批大小与跌倒轮数
batch_size = 128  # 可更改批大小
num_epoch = 65
weight_1 = 0.855  # 对标签为1的样本进行加权，设置为0.8

# 创建PyTorch数据加载器
train_data = torch.utils.data.TensorDataset(torch.Tensor(X), torch.Tensor(y), torch.Tensor(env_labels))
train_data = torch.utils.data.TensorDataset(torch.Tensor(X_combined), torch.Tensor(y_combined), torch.Tensor(env_train_reshaped))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)  # 定义批大小

class MyNet(nn.Module):
    def __init__(self, input_dim, output_dim, weight_1):
        super(MyNet, self).__init__()
        self.cnn = CNN(input_dim, output_dim)
        self.transformer = Transformer(input_dim, output_dim, nhead=12)
        self.weight_cnn = nn.Parameter(torch.ones(1))
        self.weight_transformer = nn.Parameter(torch.ones(1))

        self.classifier = nn.Linear(output_dim, output_dim)
        self.weight_1 = weight_1

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

        # 根据你的需求定义weight张量的形状
        weight = torch.ones(2)
        ground_truth_labels = ground_truth_labels.long()
        # 使用适合处理不平衡数据集的损失函数，并增加对标签为1的样本的学习权重
        if self.training:  # 处于训练模型才会执行以下代码
            loss_weight = torch.ones_like(ground_truth_labels)  # 初始化全1张量，存储每个样本的损失权重
            loss_weight[ground_truth_labels == 1] = self.weight_1  # 设置标签为1的样本损失权重
            loss_weight[ground_truth_labels == 0] = 1 - self.weight_1
            loss = F.cross_entropy(output, ground_truth_labels, weight=weight, reduction='mean')  # 计算损失权重方法，求均值
            return output
        else:
            return output

large_model = MyNet(input_dim=X.shape[1], output_dim=2, weight_1=weight_1).to(device)
optimizer = optim.Adam(large_model.parameters(), lr=0.001, weight_decay=1e-5)  # Adam优化器(最优优化器） 用于更新模型学习率
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
train_acc = 0
test_acc = 0

# 增加学习率调度器，参数需改
warmup_epochs = 5  # 预热阶段轮数
decay_epochs = 80  # 衰减阶段轮数
initial_lr = 1e-4  # 预热阶段学习率
base_lr = 1e-3  # 衰减阶段学习率
min_lr = 5e-5

# 创建一个空字典用于存储 fpr 和 tpr
fpr_dict = {}
tpr_dict = {}
# 打开文件

library = {
    'Epoch': [],
    'Loss': [],
    'F1 Score': [],
    'AUROC': [],
    'AUPC': [],
    'Sensitivity': [],
    'Specificity': [],
    'Train Acc': []
}

for epoch in range(num_epoch):
    lr = lr_scheduler(epoch, optimizer.param_groups[0]['lr'], warmup_epochs, decay_epochs, initial_lr, base_lr,min_lr)
    optimizer.param_groups[0]['lr'] = lr
    for i, (inputs, labels, groups) in enumerate(train_loader):
        inputs, labels, groups = inputs.to(device), labels.to(device), groups.to(device)
        # 正常训练过程
        large_model.train()
        optimizer.zero_grad()
        outputs = large_model(inputs, labels)
        loss = criterion(outputs.squeeze(-1), labels.long())
        #loss = irm_loss(outputs.squeeze(-1), labels, groups)  # 使用irm_loss计算模型损失值
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1} / {num_epoch}, Loss: {loss.item():.4f}')
    large_model.eval()
    with torch.no_grad():
        # 训练集
        predictions = []
        truths = []
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            output = large_model(inputs, labels.long())
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
        # 计算F1分数
        f1 = f1_score(truths, predictions)
        print(f'Epoch {epoch + 1} / {num_epoch}, F1 Score: {f1:.4f}')

        # 转换整数为张量
        predictions = [torch.tensor(output) if isinstance(output, int) else output.squeeze() for output in predictions]

        # 继续计算AUROC
        fpr, tpr, thresholds = roc_curve(truths, [output.squeeze() for output in predictions])
        auroc = roc_auc_score(truths, [output.squeeze() for output in predictions])
        print(f'Epoch {epoch + 1} / {num_epoch}, AUROC: {auroc:.4f}')

        # 计算AUPC
        aupc = auc(fpr, tpr)
        print(f'Epoch {epoch + 1} / {num_epoch}, AUPC: {aupc:.4f}')

        # 保存AUROC曲线数据
        fpr_dict[epoch + 1] = fpr
        fpr_dict[epoch + 1] = tpr

        print(
            f'Epoch {epoch + 1} / {num_epoch}, Testing: Sensitivity: {sensitivity}, Specificity: {specificity},Train Acc: {train_acc:.4f}')

        # 将数据添加到字典中
        library['Epoch'].append(epoch + 1)
        library['Loss'].append(loss.item())
        library['F1 Score'].append(f1)
        library['AUROC'].append(auroc)
        library['AUPC'].append(aupc)
        library['Sensitivity'].append(sensitivity)
        library['Specificity'].append(specificity)
        library['Train Acc'].append(train_acc)

# 创建DataFrame对象
df = pd.DataFrame(library)
# 将DataFrame保存为Excel文件
df.to_excel('evolution.xlsx', index=False)

# 创建figure1
plt.figure(num='figure1', figsize=(10, 5))
plt.plot(range(1, num_epoch + 1), train_acc_list, label='Train Accuracy')
plt.plot(range(1, num_epoch + 1), train_loss_list, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Accuracy and Loss')
plt.legend()
# 显示图形
plt.show()

# 保存大模型的参数
torch.save(large_model.state_dict(), 'large_modellv.pth')



