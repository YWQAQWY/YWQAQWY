#数值、矩阵操作
import math
import numpy as np

# 数据读取与写入make_dot
import pandas as pd
import os
import csv
import openpyxl

# 进度条
from tqdm import tqdm

# Pytorch 深度学习张量操作框架
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
# 绘制pytorch的网络
from torchviz import make_dot

# 学习曲线绘制
from torch.utils.tensorboard import SummaryWriter

file_path1 = r'C:\Users\YWQ\OneDrive\Desktop\Homework\HW1_Regression\covid.test.csv'
file_path2 = r'C:\Users\YWQ\OneDrive\Desktop\Homework\HW1_Regression\covid.train.csv'


def same_seed(seed):
    '''
    设置随机种子(便于复现)
    '''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f'Set Seed = {seed}')

def train_valid_split(data_set, valid_ratio, seed):
    '''
    数据集拆分成训练集（training set）和 验证集（validation set）
    '''
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    #这一行使用 PyTorch 的 random_split 函数将数据集 data_set 随机拆分为训练集和验证集。
    #generator 参数是一个随机数生成器，通过设置种子 seed 来确保每次拆分的结果都是一样的，从而实现可复现性。
    return np.array(train_set), np.array(valid_set)

def predict(test_loader, model, device):
    """
测试数据加载器（test_loader）：这是一个特殊的工具，用于从测试数据集中批量地加载数据。测试数据集是一组我们已经知道答案的数据，用于评估模型的性能。test_loader 通常会按批次（batch）加载数据，每个批次包含一定数量的数据样本。
模型（model）：这是一个已经训练好的模型，它已经学习了如何从输入数据中预测输出结果。模型可以是任何类型的机器学习或深度学习模型，如神经网络、决策树等。
设备（device）：这指的是模型运行的硬件平台，可以是 CPU（中央处理器）或 GPU（图形处理器）。GPU 通常在处理大量数据和进行复杂计算时比 CPU 更快。
    """
    model.eval() # 设置成eval模式.model.eval() 这一行代码就是告诉模型：“嘿，你现在进入评估模式了，不要再使用那些训练时的技巧了。”这样，我们就可以更公平、更准确地评估模型的性能了。
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds

class COVID19Dataset(Dataset):
    """
    定义一个数据集。
    x: np.ndarray  特征矩阵.
    y: np.ndarray  目标标签, 如果为None,则是预测的数据集
    如果 y 为 None，则表示这是一个用于预测的数据集。
    """
    def __init__(self,x,y=None):
        if y is None:
            self.y=y
        else:
            self.y=torch.FloatTensor(y)
        self.x=torch.FloatTensor(x)

    def __getitem__(self,idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

#构建神经网络
class My_Model(nn.Module):
    def __init__(self,in_dim):
        super(My_Model,self).__init__()#调用父类nn.module类的方法，“嘿，建筑师，我正在建造一座特别的房子，但我还是想要你提供的所有基本功能，请先帮我把这些基本功能建好。”
        #这一行开始定义一个顺序容器 self.layers，用于按顺序包含多个神经网络层。
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self,x):
        #它定义了数据在模型中的前向传播过程。
        x=self.layers(x)
        #所以，x = self.layers(x) 这一行代码的意思就是：
        # “把原材料 x 通过加工步骤 self.layers 进行加工，得到加工后的结果，还是叫它 x。”
        # 这样，x 就不再是原始的原材料，而是经过一系列加工步骤后的半成品了。
        x = x.squeeze(1)  # (B, 1) -> (B)
        return x


def select_feat(train_data, valid_data, test_data, select_all=True):
    '''
    特征选择
    选择较好的特征用来拟合回归模型
    '''
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]
    #这一行从训练数据和验证数据中提取目标标签(通常位于数据的最后一列)
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1], test_data
    #这一行从训练数据和验证数据中提取特征（通常位于数据的除最后一列之外的所有列），并将测试数据赋值给
    #raw_x_test。
    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0,1,2,3,4] # TODO: 选择需要的特征 ，这部分可以调研一些特征选择的方法并完善.

    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid

def trainer (train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean') # 损失函数的定义 这一行定义了损失函数 criterion，
    # 这里使用的是均方误差损失（MSELoss），它是回归问题中常用的损失函数。
    # reduction='mean' 表示损失值会取平均。
    optimizer = torch.optim.SGD(model.parameters(),lr=config['learning_rate'],momentum=0.9)
    writer = SummaryWriter()

    if not os.path.isdir('./models'):
        # 创建文件夹-用于存储模型
        os.mkdir('./models')

    #初始化训练参数
    """
    best_loss：最佳验证损失，初始设为无穷大。
    step：当前训练步数，初始为0。
    early_stop_count：提前停止计数器，初始为0。
    """
    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        # 训练模式model.train() 将模型设置为训练模式，loss_record 用于记录每个批次的损失值。
        model.train()
        loss_record = []

        # 使用 tqdm 显示训练进度,
        # tqdm可以帮助我们显示训练的进度
        # set_description 设置了进度条的描述信息，显示当前是第几个训练周期。
        train_pbar = tqdm(train_loader, position=0, leave=True)
        train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')

        #训练过程
        for x, y in train_pbar:
            optimizer.zero_grad()  # 将梯度置0.
            x, y = x.to(device), y.to(device)  # 将数据一到相应的存储位置(CPU/GPU)
            pred = model(x) #模型预测
            loss = criterion(pred, y)
            loss.backward()  # 反向传播 计算梯度.
            optimizer.step()  # 更新网络参数
            step += 1
            loss_record.append(loss.detach().item()) # 记录当前批次的损失值。
            train_pbar.set_postfix({'loss': loss.detach().item()}) #更新进度条显示的损失值

    mean_train_loss = sum(loss_record) / len(loss_record)# 计算出损失函数的平均值
    # 每个epoch，在tensorboard中记录训练的损失（后面可以展示出来）
    writer.add_scalar('loss/train', mean_train_loss, step)

    # 验证模式
    model.eval()  # 将模型设置成 evaluation 模式。
    loss_record = []
    for x, y in valid_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad(): # 禁用梯度计算，减少内存消耗
            pred = model(x)
            loss = criterion(pred, y)
        loss_record.append(loss.detach().item())
    mean_valid_loss = sum(loss_record) / len(loss_record)#计算平均
    writer.add_scalar('loss/valid', mean_valid_loss, step)

    print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')

    #保存最佳模型
    if mean_valid_loss < best_loss:
        best_loss = mean_valid_loss
        torch.save(model.state_dict(), config['save_path'])  # 保存模型
        print('Saving model with loss {:.3f}...'.format(best_loss))
        early_stop_count = 0 #重置提前停止计数器
    else:
        early_stop_count += 1

    #提前停止
    #因为模型在验证集上的性能没有提升，继续训练可能会导致过拟合。
    if early_stop_count >= config['early_stop']:
        print('\nModel is not improving, so we halt the training session.')
        return


device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 20051023,      # 随机种子，可以自己填写. :)
    'select_all': True,   # 是否选择全部的特征
    'valid_ratio': 0.2,   # 验证集大小(validation_size) = 训练集大小(train_size) * 验证数据占比(valid_ratio)
    'n_epochs': 3000,     # 数据遍历训练次数
    'batch_size': 256,
    'learning_rate': 1e-5,
    'early_stop': 400,    # 如果early_stop轮损失没有下降就停止训练.
    'save_path': './models/model.ckpt'  # 模型存储的位置
}

same_seed(config['seed'])

pd.set_option('display.max_column', 200) # 设置显示数据的列数
train_df, test_df = pd.read_csv(file_path2), pd.read_csv(file_path1)
train_data, test_data = train_df.values, test_df.values
del train_df, test_df # 删除数据减少内存占用
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])
train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
                                            COVID19Dataset(x_valid, y_valid), \
                                            COVID19Dataset(x_test)

# 使用Pytorch中Dataloader类按照Batch将数据集加载
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

model = My_Model(in_dim=x_train.shape[1]).to(device) # 将模型和训练数据放在相同的存储位置(CPU/GPU)
trainer(train_loader, valid_loader, model, config, device)

def save_pred(preds, file):
    ''' 将模型保存到指定位置'''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

model = My_Model(in_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader, model, device)
save_pred(preds, 'pred.csv')