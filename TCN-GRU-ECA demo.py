import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from TCN import TemporalConvNet
from SE_or_ECA_or_CBAM import ECALayer
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载与预处理
df = pd.read_csv('weather.csv')
features = df.values[:, :]
labels = df.iloc[:, -1].values.reshape(-1, 1)  # 注意：最后一列为labels

# 数据归一化
# LSTM / GRU常用minmax归一化
scaler_labels = MinMaxScaler()
scaler_features = MinMaxScaler()
scaler_labels.fit(labels)
features = scaler_features.fit_transform(features)


# 创建时间序列数据集
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, -1]  # 注意：预测最后一列的下一个时间点
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# 滑窗长度
look_back = 10
X, y = create_sequences(features, look_back)

# 将数据划分为训练集和测试集
train_size = int(len(df) * 0.7)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 转换为 PyTorch 张量
# 兼容特征为单维的情况
if len(X_train.shape) < 3:
    # 如果特征为单维，则需额外添加一个维度
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(device)
else:
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class TCN_GRU_ECA(nn.Module):
    def __init__(self, input_dim, hidden_dim, TCN_layers, output_dim, bidirectional, kernel_size=4):
        super(TCN_GRU_ECA, self).__init__()
        self.tcn = TemporalConvNet(input_dim, [hidden_dim] * TCN_layers, kernel_size)
        self.gru = self.gru = nn.GRU(hidden_dim, hidden_dim, bidirectional=bidirectional, batch_first=True)
        self.eca = ECALayer(hidden_dim * 2) if bidirectional else ECALayer(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) if bidirectional else nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq, feature)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        # SE/ECA/CBAM 的期望输入为 (batch, feature, seq)
        x = x.transpose(1, 2)
        x = self.eca(x)
        x = x.transpose(1, 2)
        x = x[:, -1, :]
        x = self.fc(x)
        return x.squeeze()


# 超参数
input_dim = features.shape[1]
hidden_dim = 64
TCN_layers = 3
output_dim = 1
num_epochs = 50
bidirectional = False

model = TCN_GRU_ECA(input_dim, hidden_dim, TCN_layers, output_dim, bidirectional).to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
model.train()
train_loss = []
test_loss = []

for epoch in range(num_epochs):
    #  训练
    model.train()
    epoch_train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * X_batch.size(0)
    avg_train_loss = epoch_train_loss / len(train_dataset)
    #  验证
    model.eval()
    epoch_test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            epoch_test_loss += loss.item() * X_batch.size(0)
    avg_test_loss = epoch_test_loss / len(test_dataset)
    # 记录打印
    train_loss.append(avg_train_loss)
    test_loss.append(avg_test_loss)
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss}, Test Loss: {avg_test_loss}')


# 指标函数
def r2_score(y_pred, y_test):
    y_pred = y_pred.view(-1)
    y_test = y_test.view(-1)
    ss_res = torch.sum((y_test - y_pred) ** 2)
    ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def rmse(y_pred, y_test):
    y_pred = y_pred.view(-1)
    y_test = y_test.view(-1)
    mse = torch.mean((y_test - y_pred) ** 2)
    return torch.sqrt(mse)


def mae(y_pred, y_test):
    y_pred = y_pred.view(-1)
    y_test = y_test.view(-1)
    return torch.mean(torch.abs(y_test - y_pred))


# 模型测试
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    MSE = criterion(y_pred, y_test)
    R2 = r2_score(y_pred, y_test)
    RMSE = rmse(y_pred, y_test)
    MAE = mae(y_pred, y_test)
    print('--------------------------------------------')
    print(f'MSE: {MSE.item()}')
    print(f'R2:{R2.item()}')
    print(f'RMSE:{RMSE.item()}')
    print(f'MAE:{MAE.item()}')
    print('--------------------------------------------')

# 转换回原始数据尺度
y_pred = y_pred.cpu().numpy()
y_test = y_test.cpu().numpy()
y_pred = y_pred.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_pred = scaler_labels.inverse_transform(y_pred)
y_test = scaler_labels.inverse_transform(y_test)

# 画图
plt.plot(range(len(train_loss)), train_loss, 'k')
plt.plot(range(len(test_loss)), test_loss, 'r')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.legend(('Training set', 'Test set'))
plt.show()

plt.plot(range(len(y_pred)), y_pred, 'r')
plt.plot(range(len(y_test)), y_test, 'b')
plt.legend(('Predicted', 'Actual'))
plt.show()

# df = pd.DataFrame(y_pred)
# df.to_csv("Predicted.csv", index=False, header=False)