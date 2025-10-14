import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from functions2 import lr_scheduler, CNN, Transformer, dynamic_weight_adjustment, compute_accuracy
from sklearn.metrics import roc_auc_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score
# 使用自建数据集5-折交叉验证的结果
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# 定义设备
device = 'cpu'
num_epochs_finetune = 100

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
optimizer_low_rank = optim.SGD([  # 定义优化器
    {'params': model2.cnn.parameters(), 'lr': 0.001},
    {'params': model2.transformer.parameters(), 'lr': 0.001},
    {'params': model2.classifier.parameters(), 'lr': 0.01}
], momentum=0.9)

# 定义交叉验证的批次
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# 计算准确率
def accuracy(predictions, truths):
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == truths[i]:
            correct += 1
    return correct / len(predictions)


# 用于存储不同折叠的训练/验证准确率、损失
train_acc_list = []
train_loss_list = []
val_acc_list = []
val_loss_list = []
train_recall_list = []
val_recall_list = []

# 增加学习率调度器，参数需改
warmup_epochs = 5  # 预热阶段轮数
decay_epochs = 20  # 衰减阶段轮数
initial_lr = 1e-4  # 预热阶段学习率
base_lr = 1e-3  # 衰减阶段学习率
min_lr = 5e-5

# 数据预处理
scaler = StandardScaler()  # 归一化处理
scaler.fit(new_X)  # 先进行fit操作
new_X = scaler.transform(new_X)

# 第一步：将数据集划分为训练集和临时集（临时集包含测试集和验证集）
new_X_train, new_X_temp, new_y_train, new_y_temp, new_env_train, new_env_temp = train_test_split(
    new_X, new_y, new_env, test_size=0.3, random_state=42)

# 第二步：将临时集划分为测试集和验证集
new_X_test, new_X_val, new_y_test, new_y_val, new_env_test, new_env_val = train_test_split(
    new_X_temp, new_y_temp, new_env_temp, test_size=0.2 / 0.3, random_state=42)

test_data = torch.utils.data.TensorDataset(torch.Tensor(new_X_test), torch.Tensor(new_y_test),
                                           torch.Tensor(new_env_test))
new_test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

# 进行5折交叉验证
for fold, (train_idx, val_idx) in enumerate(kf.split(new_X_train)):
    print(f"\nFold {fold + 1}/{5}")

    # 获取训练集和验证集数据
    X_train_fold = new_X_train[train_idx]
    y_train_fold = new_y_train[train_idx]
    X_val_fold = new_X_train[val_idx]
    y_val_fold = new_y_train[val_idx]

    # 创建数据加载器
    train_data = torch.utils.data.TensorDataset(torch.Tensor(X_train_fold), torch.Tensor(y_train_fold),
                                                torch.Tensor(new_env_train[train_idx]))
    val_data = torch.utils.data.TensorDataset(torch.Tensor(X_val_fold), torch.Tensor(y_val_fold),
                                              torch.Tensor(new_env_train[val_idx]))

    new_train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    new_val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=True)

    # 训练和验证模型
    for epoch in range(num_epochs_finetune):
        lr = lr_scheduler(epoch, optimizer_low_rank.param_groups[0]['lr'], warmup_epochs, decay_epochs, initial_lr,
                          base_lr, min_lr)
        optimizer_low_rank.param_groups[0]['lr'] = lr

        model2.train()
        for i, (inputs, labels, groups) in enumerate(new_train_loader):
            inputs, labels, groups = inputs.to(device), labels.to(device), groups.to(device)
            optimizer_low_rank.zero_grad()
            outputs = model2(inputs, labels)
            loss = F.cross_entropy(outputs.squeeze(-1), labels.long())
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

            train_acc = accuracy(predictions, truths)
            train_recall = recall_score(truths, predictions, average='macro')
            train_loss = loss.item()

            # 验证集
            predictions = []
            truths = []
            for inputs, labels, _ in new_val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model2(inputs, labels)
                _, predicted = torch.max(output, 1)
                predictions += predicted.tolist()
                truths += labels.tolist()

            val_acc = accuracy(predictions, truths)
            val_loss = loss.item()
            val_recall = recall_score(truths, predictions, average='macro')

            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            train_recall_list.append(train_recall)  # ✅ 新增
            val_recall_list.append(val_recall)  # ✅ 新增

            # ---------- 输出 ----------
            print(
                f"Epoch {epoch + 1}/{num_epochs_finetune}, "
                f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
                f"Train Recall: {train_recall:.4f}, Val Recall: {val_recall:.4f}, "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )


# 绘制折线图
plt.figure(figsize=(10, 5))  # 设置图形窗口大小
# 绘制训练集和验证集准确率折线
plt.plot(range(1, num_epochs_finetune * 5 + 1), train_acc_list, label='Train Accuracy')
plt.plot(range(1, num_epochs_finetune * 5 + 1), val_acc_list, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy (5-Fold Cross Validation)')
plt.legend()

plt.figure(figsize=(10, 5))  # 设置图形窗口大小


# ========================
# 绘制 5 折交叉验证结果图
# ========================

# 将列表转换为 numpy 数组以便 reshape
train_acc_array = np.array(train_acc_list).reshape(5, num_epochs_finetune)
val_acc_array = np.array(val_acc_list).reshape(5, num_epochs_finetune)
train_loss_array = np.array(train_loss_list).reshape(5, num_epochs_finetune)
val_loss_array = np.array(val_loss_list).reshape(5, num_epochs_finetune)

# 计算平均曲线
mean_train_acc = np.mean(train_acc_array, axis=0)
mean_val_acc = np.mean(val_acc_array, axis=0)
mean_train_loss = np.mean(train_loss_array, axis=0)
mean_val_loss = np.mean(val_loss_array, axis=0)

# ---- 绘制准确率曲线 ----
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.plot(range(1, num_epochs_finetune + 1), train_acc_array[i], linestyle='--', alpha=0.5, label=f'Train Fold {i+1}')
    plt.plot(range(1, num_epochs_finetune + 1), val_acc_array[i], alpha=0.5, label=f'Val Fold {i+1}')

plt.plot(range(1, num_epochs_finetune + 1), mean_train_acc, color='blue', linewidth=2.5, label='Mean Train Accuracy')
plt.plot(range(1, num_epochs_finetune + 1), mean_val_acc, color='orange', linewidth=2.5, label='Mean Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy per Fold (5-Fold Cross Validation)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- 绘制损失曲线 ----
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.plot(range(1, num_epochs_finetune + 1), train_loss_array[i], linestyle='--', alpha=0.5, label=f'Train Loss Fold {i+1}')
    plt.plot(range(1, num_epochs_finetune + 1), val_loss_array[i], alpha=0.5, label=f'Val Loss Fold {i+1}')

plt.plot(range(1, num_epochs_finetune + 1), mean_train_loss, color='blue', linewidth=2.5, label='Mean Train Loss')
plt.plot(range(1, num_epochs_finetune + 1), mean_val_loss, color='orange', linewidth=2.5, label='Mean Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Fold (5-Fold Cross Validation)')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()
