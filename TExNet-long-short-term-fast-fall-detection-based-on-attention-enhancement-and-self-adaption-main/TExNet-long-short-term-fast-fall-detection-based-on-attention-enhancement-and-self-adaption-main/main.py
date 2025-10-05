import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from functions import self_sliding, PositionalEmbedding, lr_scheduler, CNN, Transformer
from sklearn.metrics import multilabel_confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 用于训练的第二个数据集数据 自建数据集不经过微调
data = pd.read_excel('自建数据-平衡后的数据.xlsx')  # 引入新的数据文件
X = data.iloc[:, 1:13].values
y = data.iloc[:, 13].values

# 数据预处理
scaler = StandardScaler()  # 归一化处理
X = scaler.fit_transform(X)

# 划分数据集 random_state指定固定的随机状态，保证每次数据的拆分结果一样，模型可重复
# 先将数据划分为训练集和临时集（包含测试集和验证集），临时集占比 0.3
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# 再将临时集划分为测试集和验证集，测试集占比 2/3（即总数据的 0.2），验证集占比 1/3（即总数据的 0.1）
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

# 划分滑动窗口
X_train_sliding = self_sliding(X_train, window_width=370, stride=50)
y_train_sliding = self_sliding(y_train, window_width=370, stride=50)

X_test_sliding = self_sliding(X_test, window_width=370, stride=50)
y_test_sliding = self_sliding(y_test, window_width=370, stride=50)

X_val_sliding = self_sliding(X_val, window_width=370, stride=50)
y_val_sliding = self_sliding(y_val, window_width=370, stride=50)

num_samples_train = X_train_sliding.shape[0]  # 训练集样本数量
num_windows_train = X_train_sliding.shape[1]  # 训练集窗口数量

X_train_reshaped = np.reshape(X_train_sliding, (num_samples_train * num_windows_train, 12))  # 需要和输入特征个数相同
y_train_reshaped = np.reshape(y_train_sliding, (num_samples_train * num_windows_train,))

num_samples_test = X_test_sliding.shape[0]  # 测试集样本数量
num_windows_test = X_test_sliding.shape[1]  # 测试集窗口数量

X_test_reshaped = np.reshape(X_test_sliding, (num_samples_test * num_windows_test, 12))
y_test_reshaped = np.reshape(y_test_sliding, (num_samples_test * num_windows_test,))

num_samples_val = X_val_sliding.shape[0]  # 验证集样本数量
num_windows_val = X_val_sliding.shape[1]  # 验证集窗口数量

X_val_reshaped = np.reshape(X_val_sliding, (num_samples_val * num_windows_val, 12))
y_val_reshaped = np.reshape(y_val_sliding, (num_samples_val * num_windows_val,))

X_combined = np.concatenate((X_train_reshaped, X_test_reshaped, X_val_reshaped), axis=0)
y_combined = np.concatenate((y_train_reshaped, y_test_reshaped, y_val_reshaped), axis=0)

# 重新划分数据集
X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y_combined, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

# 规定批大小与跌倒轮数
batch_size = 128  # 可更改批大小
num_epoch = 100

# 创建PyTorch数据加载器
train_data = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
test_data = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
val_data = torch.utils.data.TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)  # 定义批大小
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建Transformer模型
class Mynet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Mynet, self).__init__()
        self.cnn = CNN(input_dim, output_dim)
        self.transformer = Transformer(input_dim, output_dim, nhead=12)  # 修改输入维度为input_dim-1
        self.classifier = nn.Linear(output_dim*2, output_dim)  # 修改分类器的输入维度为output_dim*2，输出维度为output_dim

    def forward(self, x):
        cnn_output = self.cnn(x)
        transformer_output = self.transformer(x)
        combined_output = torch.cat((cnn_output, transformer_output), dim=1)
        output = self.classifier(combined_output)
        return output

model = Mynet(input_dim=X.shape[1], output_dim=3).to(device)  # input_dim 输入维度为X列数，输出维度为分类数
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Adam优化器(最优优化器） 用于更新模型学习率
# weight_decay添加正则化，相当于L2正则化（可改参数）
criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数，用于比较模型预测值和真实值的差异

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
decay_epochs = 5  # 衰减阶段轮数
initial_lr = 1e-4  # 预热阶段学习率
base_lr = 1e-3  # 衰减阶段学习率
min_lr = 5e-5

for epoch in range(num_epoch):
    lr = lr_scheduler(epoch, optimizer.param_groups[0]['lr'], warmup_epochs, decay_epochs, initial_lr, base_lr, min_lr)
    optimizer.param_groups[0]['lr'] = lr

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())  # 直接传入整数标签，无需转换为长整型
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        # 训练集
        predictions = []
        truths = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            output = model(inputs)
            _, predicted = torch.max(output, 1)  # 使用torch.max获取每个样本的预测类别
            predictions += predicted.tolist()
            truths += labels.tolist()

        confusion_mat = multilabel_confusion_matrix(truths, predictions)
        tn = confusion_mat[:, 0, 0]
        fp = confusion_mat[:, 0, 1]
        fn = confusion_mat[:, 1, 0]
        tp = confusion_mat[:, 1, 1]

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        # 每一轮训练中计算准确率和损失值
        train_predictions = torch.argmax(output, dim=1)
        # 计算训练集准确率
        train_acc = accuracy(predictions, truths)
        train_loss = loss.item()

        # 保存准确率和损失值
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        # 测试集
        predictions = []
        truths = []
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            output = model(inputs)

            # 将输出结果用 softmax 函数转换为概率
            probabilities = torch.nn.functional.softmax(output, dim=1)

            predicted = torch.argmax(probabilities, dim=1)

            predictions += predicted.tolist()
            truths += labels.tolist()

        confusion_mat = multilabel_confusion_matrix(truths, predictions)
        tn = confusion_mat[:, 0, 0]
        fp = confusion_mat[:, 0, 1]
        fn = confusion_mat[:, 1, 0]
        tp = confusion_mat[:, 1, 1]

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        # 计算测试集准确率
        test_acc = accuracy(predictions, truths)

        # 验证集
        predictions = []
        truths = []
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            output = model(inputs)

            # 将输出结果用 softmax 函数转换为概率
            probabilities = torch.nn.functional.softmax(output, dim=1)

            predicted = torch.argmax(probabilities, dim=1)

            predictions += predicted.tolist()
            truths += labels.tolist()

        confusion_mat = multilabel_confusion_matrix(truths, predictions)
        tn = confusion_mat[:, 0, 0]
        fp = confusion_mat[:, 0, 1]
        fn = confusion_mat[:, 1, 0]
        tp = confusion_mat[:, 1, 1]

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        # 计算验证集准确率
        val_acc = accuracy(predictions, truths)

        print(
            f'Epoch {epoch + 1} / {num_epoch}, Testing: Sensitivity: {sensitivity}, Specificity: {specificity}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Val Acc: {val_acc:.4f}')
        #  sensitivity 灵敏度（召回率）：正样本与正预测之比    specificity特异性：实际负样本与负样本之比  AUC: 显示分类器效果，值越接近1效果越好

# 绘制折线图
plt.figure(figsize=(10, 5))  # 设置图形窗口大小

# 绘制训练集准确率折线
plt.plot(range(1, num_epoch + 1), train_acc_list, label='Train Accuracy')
plt.plot(range(1, num_epoch + 1), train_loss_list, label='Train Loss')
plt.plot(range(1, num_epoch + 1), test_acc_list, label='Test Accuracy')
plt.plot(range(1, num_epoch + 1), test_loss_list, label='Test Loss')
plt.plot(range(1, num_epoch + 1), val_acc_list, label='Validation Accuracy')
plt.plot(range(1, num_epoch + 1), val_loss_list, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training, Testing and Validation Accuracy and Loss')
plt.legend()
# 显示图形
plt.show()