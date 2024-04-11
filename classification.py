import random

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.optim import RMSprop

from model import LSTMModel

# from torch import nn

path = './data/knn_clean_data.csv'
df = pd.read_csv(path)

moods = df[['ID', 'time', 'mood']]

moods.sort_values(by=['time'], inplace=True)

moods['time'] = pd.to_datetime(moods['time'])
moods['time'] -= min(moods['time'])
moods['time'] = moods['time'].dt.days
# print(moods['time'])

moods = pd.get_dummies(moods, columns=['ID'])
moods['mood'] = moods['mood'].round().astype(int)
print(moods['mood'].value_counts())

# print(moods)
# print(moods.shape)
# X = moods.drop(columns=['mood']).values  # 特征
# y = moods['mood'].values  # 目标变量

# print(X)

# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据转换为PyTorch张量
# X_train = torch.tensor(X_train, dtype=torch.float32)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32)
# y_test = torch.tensor(y_test, dtype=torch.float32)

# print(X_train[0])
# dataset = TensorDataset(X_train, y_train)
hidden_size = 128  # 隐藏层维度
num_layers = 2  # LSTM层数
output_size = 10  # 输出维度
num_epochs = 5000  # 迭代次数
learning_rate = 1e-2  # 学习率
max_grad_norm = 5.0

seq_length = 7
batch_size = 128
step = 1
# print(moods)
ids = df['ID'].unique()
# print(ids)
id_moods = []
for uid in ids:
    # print(uid)
    uid = 'ID_' + uid
    id_mood = moods[moods[uid] == 1]
    id_moods.append(id_mood)
    # print(len(id_mood))
    # print(uid)
    # print(moods[moods[uid] == 1])
# print(id_moods)
id_moods[0] = id_moods[0][2:]
dataset = []
for id_mood in id_moods:
    # 最后一行为label
    # print(len(id_mood))
    id_dataset = [id_mood[i:i + seq_length + 1] for i in range(0, len(id_mood) - seq_length, step)]
    id_dataset = [df.to_numpy() for df in id_dataset]
    # print(len(id_dataset))
    # print(id_dataset)
    dataset += id_dataset

print('dataset len:', len(dataset))
# dataset = np.array(dataset)
# print(X)
# y = [df[i:i+1] for i in range(seq_length, len(moods), seq_length)]
# print(y)
# print(len(X), len(y))
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False
random_indices = random.sample(range(len(dataset)), k=int(len(dataset) / 5))
train_set = []
test_set = []

for i in range(len(dataset)):
    if i in random_indices:
        test_set.append(dataset[i])
    else:
        train_set.append(dataset[i])
# print(train_set)
print(len(train_set))
# print(test_set)
print(len(test_set))
# dataset -= train_set
# print(len(dataset))
X_train = [element[:seq_length] for element in train_set]
# print(X_train)
y_train = [element[seq_length:][0][1:2] for element in train_set]
# y_train = [element[seq_length:] for element in train_set]
print(y_train)
print(len(y_train))
X_test = [element[:seq_length] for element in test_set]
y_test = [element[seq_length:][0][1:2] for element in test_set]
# y_test = [element[seq_length:] for element in test_set]
# print(y_train[0][0][1])
# X_train = [element[:seq_length] for element in train_set]
# y_train = [element[seq_length:] for element in train_set]
# print(y_train)
# print(train_set[0].shape)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
# 定义参数
input_size = X_train.shape[-1]  # 输入维度
# print(X_train.shape, input_size)


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建模型并将其移动到设备上
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

# 定义损失函数和优化器
criterion = CrossEntropyLoss()
# optimizer = Adam(model.parameters(), lr=learning_rate)
optimizer = RMSprop(model.parameters(), lr=learning_rate)

loss_history = []
accuracy_history = []
# 训练模型
batch_num = len(X_train) // batch_size

for epoch in range(num_epochs):
    total_loss = 0
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i + batch_size].to(device)
        targets = y_train[i:i + batch_size].to(device)
        # print(targets)
        if len(targets) < batch_size:
            continue
        # 前向传播
        outputs = model(inputs)
        # print(outputs)
        # print(outputs.shape)
        # print(targets.shape)
        loss = criterion(outputs, targets.squeeze().long())
        total_loss += loss.item()
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
    total_loss = total_loss / batch_num
    loss_history.append(total_loss)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}')

        with torch.no_grad():
            correct = 0
            total = 0
            for i in range(0, len(X_test), batch_size):
                inputs = X_test[i:i + batch_size].to(device)
                targets = y_test[i:i + batch_size].to(device)
                targets = targets.transpose(0, 1)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                # print(predicted)
                # print(targets)
                # print(targets.size(1))
                total += targets.size(1)
                correct_matrix = (predicted == targets)
                # print(correct_matrix.sum())
                correct += (predicted == targets).sum().item()
                # print(total, correct)

            print(f'Accuracy on test set: {100 * correct / total:.2f}%')
            accuracy_history.append(100 * correct / total)
    # 模型评估

plt.figure(figsize=(10, 5))
plt.plot(loss_history, 'r', label='Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 绘制准确率图像
plt.figure(figsize=(10, 5))
plt.plot(accuracy_history, 'b', label='Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
