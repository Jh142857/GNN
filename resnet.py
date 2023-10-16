from json import load
from platform import node
from statistics import mode
from utils import DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T

import numpy as np
from torch_geometric.data import Data
import os

from torch.utils.tensorboard import SummaryWriter


# 设置gpu设备
device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
print(f'设备：{device}')

# 路径准备
path='./datasets/632d74d4e2843a53167ee9a1-momodel/' #数据保存路径
save_dir='./results/mlp' #模型保存路径
dataset_name='DGraph'

# 参数
mlp_parameters = {
    'lr': 0.01
    , 'num_layers': 4
    , 'hidden_channels': 128
    , 'dropout': 0.0
    , 'batchnorm': False
    , 'weight_decay': 5e-7
}
epochs = 500
log_steps =10 # log记录周期
para_dict = mlp_parameters
# 评价指标
eval_metric = 'auc'  #使用AUC衡量指标
evaluator = Evaluator(eval_metric)

# ResNet
class ResNet(nn.Module):
    """
    __init__
        block: 堆叠的基本模块
        block_num: 基本模块堆叠个数,是一个list,对于resnet50=[3,4,6,3]
        num_classes: 全连接之后的分类特征维度
        
    _make_layer
        block: 堆叠的基本模块
        channel: 每个stage中堆叠模块的第一个卷积的卷积核个数，对resnet50分别是:64,128,256,512
        block_num: 当期stage堆叠block个数
        stride: 默认卷积步长
    """
    def __init__(self, block, block_num, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channel = 64    # conv1的输出维度
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)     # H/2,W/2。C:3->64
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)     # H/2,W/2。C不变
        self.layer1 = self._make_layer(block=block, channel=64, block_num=block_num[0], stride=1)   # H,W不变。downsample控制的shortcut，out_channel=64x4=256
        self.layer2 = self._make_layer(block=block, channel=128, block_num=block_num[1], stride=2)  # H/2, W/2。downsample控制的shortcut，out_channel=128x4=512
        self.layer3 = self._make_layer(block=block, channel=256, block_num=block_num[2], stride=2)  # H/2, W/2。downsample控制的shortcut，out_channel=256x4=1024
        self.layer4 = self._make_layer(block=block, channel=512, block_num=block_num[3], stride=2)  # H/2, W/2。downsample控制的shortcut，out_channel=512x4=2048

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.fc = nn.Linear(in_features=512*block.expansion, out_features=num_classes)

        for m in self.modules():    # 权重初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None   # 用于控制shorcut路的
        if stride != 1 or self.in_channel != channel*block.expansion:   # 对resnet50：conv2中特征图尺寸H,W不需要下采样/2，但是通道数x4，因此shortcut通道数也需要x4。对其余conv3,4,5，既要特征图尺寸H,W/2，又要shortcut维度x4
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=channel*block.expansion, kernel_size=1, stride=stride, bias=False), # out_channels决定输出通道数x4，stride决定特征图尺寸H,W/2
                nn.BatchNorm2d(num_features=channel*block.expansion))

        layers = []  # 每一个convi_x的结构保存在一个layers列表中，i={2,3,4,5}
        layers.append(block(in_channel=self.in_channel, out_channel=channel, downsample=downsample, stride=stride)) # 定义convi_x中的第一个残差块，只有第一个需要设置downsample和stride
        self.in_channel = channel*block.expansion   # 在下一次调用_make_layer函数的时候，self.in_channel已经x4

        for _ in range(1, block_num):  # 通过循环堆叠其余残差块(堆叠了剩余的block_num-1个)
            layers.append(block(in_channel=self.in_channel, out_channel=channel))

        return nn.Sequential(*layers)   # '*'的作用是将list转换为非关键字参数传入

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 多层感知机
class MLP(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , batchnorm=True):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        # 输入线性层
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        # 隐藏线性层
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        # 输出线性层
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        # dropout层
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x):
        # 除了输出层，激活函数均是relu
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # 最后一层的输出是softmax
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=-1)


def precess_data():

    # 数据集为InMemoryDataset继承而来的类型，转化成稀疏矩阵
    dataset = DGraphFin(root=path, name=dataset_name, transform=T.ToSparseTensor())

    # 开始时标签是1
    nlabels = dataset.num_classes
    # print(nlabels)
    if dataset_name in ['DGraph']:
        nlabels = 2    #本实验中仅需预测类0和类1

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric() #将有向图转化为无向图

    # 归一化处理
    if dataset_name in ['DGraph']:
        x = data.x
        x = (x - x.mean(0)) / x.std(0)
        data.x = x
    # 降维
    if data.y.dim() == 2:
        data.y = data.y.squeeze(1)

    # 转化成GPU类型张量
    data.x = data.x.to(device)
    data.y = data.y.to(device)

    split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}  #划分训练集，验证集

    train_idx = split_idx['train']
    # result_dir = prepare_folder(dataset_name,'mlp')

    # print(data)
    # print(data.x.shape)  #feature
    # print(data.y.shape)  #label
    return data, split_idx, train_idx, nlabels


def init_model(data, nlabels=2, if_train=True):
    # 初始化模型
    model_para = mlp_parameters.copy()
    model_para.pop('lr')
    model_para.pop('weight_decay')
    model = MLP(in_channels=data.x.size(-1), out_channels=nlabels, **model_para).to(device)
    if if_train:
        print(f'模型：MLP')
        print(f'模型参数总量：{sum(p.numel() for p in model.parameters())}')  #模型总参数量
    return model


# 一个epoch训练
def train(model, data, train_idx, optimizer):
     # data.y is labels of shape (N, ) 
    model.train()

    optimizer.zero_grad()
    
    out = model(data.x[train_idx])

    loss = F.nll_loss(out, data.y[train_idx])
    loss.backward()
    optimizer.step()

    # 将此张量的值作为标准 Python 数字返回
    return loss.item()


def test(model, data, split_idx, evaluator):
    # data.y is labels of shape (N, )
    with torch.no_grad():
        model.eval()

        losses, eval_results = dict(), dict()
        for key in ['train', 'valid']:
            # 返回的是节点
            node_id = split_idx[key]
            
            out = model(data.x[node_id])
            y_pred = out.exp()  # (N,num_classes)
            
            # 负对数似然损失
            losses[key] = F.nll_loss(out, data.y[node_id]).item()
            eval_results[key] = evaluator.eval(data.y[node_id], y_pred)[eval_metric]

    return eval_results, losses, y_pred



def train_multiple(model, data, train_idx, split_idx, save_dir, resume_epoch=0):
    model.reset_parameters()
    if resume_epoch and resume_epoch < epochs:
        print(f"从{resume_epoch}个epoch恢复训练")
        resume_pt = save_dir + f'/model-{mlp_parameters["num_layers"]}layers-{resume_epoch}epochs.pt'
        model.load_state_dict(torch.load(resume_pt))
    optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['weight_decay'])
    best_valid = 0
    min_valid_loss = 1e8

    if not resume_epoch:
        print("模型开始训练")
    writer = SummaryWriter("./results/tb_results")
    for epoch in range(resume_epoch + 1,epochs + 1):
        loss = train(model, data, train_idx, optimizer)
        eval_results, losses, out = test(model, data, split_idx, evaluator)
        train_eval, valid_eval = eval_results['train'], eval_results['valid']
        train_loss, valid_loss = losses['train'], losses['valid']
        writer.add_scalar('total_loss', loss, epoch)
        writer.add_scalar('train_AUC', train_eval, epoch)
        writer.add_scalar('vaild_AUC', valid_eval, epoch)

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), save_dir+f'/model-{mlp_parameters["num_layers"]}layers-{epochs}epochs.pt') #将表现最好的模型保存

        if epoch % log_steps == 0:
            print(f'Epoch: {epoch:02d}, '
                f'Loss: {loss:.4f}, '
                f'Train: {100 * train_eval:.3f}, ' # 我们将AUC值乘上100，使其在0-100的区间内
                f'Valid: {100 * valid_eval:.3f} ')



def predict(model, data, node_id):
    """
    加载模型和模型预测
    :param node_id: int, 需要进行预测节点的下标
    :return: tensor, 类0以及类1的概率, torch.size[1,2]
    """
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    with torch.no_grad():
        model.eval()
        out = model(data.x[node_id])
        y_pred = out.exp()  # (N,num_classes)
        
    return y_pred


def train_all(resume_epoch=0):
    data, split_idx, train_idx, nlabels = precess_data()
    model = init_model(data, nlabels=nlabels)
    train_multiple(model, data, train_idx, split_idx, save_dir, resume_epoch=resume_epoch)


def inference():
    data, split_idx, train_idx, nlabels = precess_data()
    model = init_model(data, nlabels=nlabels, if_train=False)
    model.load_state_dict(torch.load(save_dir+f'/model-{mlp_parameters["num_layers"]}layers-{epochs}epochs.pt')) #载入验证集上表现最好的模型

    dic={0:"正常用户",1:"欺诈用户"}
    node_idx = 0
    y_pred = predict(model, data, node_idx)
    print(y_pred)
    print(f'节点 {node_idx} 预测对应的标签为:{torch.argmax(y_pred)}, 为{dic[torch.argmax(y_pred).item()]}。')

    node_idx = 1
    y_pred = predict(model, data, node_idx)
    print(y_pred)
    print(f'节点 {node_idx} 预测对应的标签为:{torch.argmax(y_pred)}, 为{dic[torch.argmax(y_pred).item()]}。')


def get_score():
    """
    得到测试集评分
    """
    data, split_idx, train_idx, nlabels = precess_data()
    model = init_model(data, nlabels=nlabels, if_train=False)
    model.load_state_dict(torch.load(save_dir+'/model.pt')) #载入验证集上表现最好的模型

    node_ids = split_idx['test']
    y_pred = predict(model, data, node_ids)
    # print(y_pred, y_pred.shape)
    # print(data.y[node_ids], data.y[node_ids].shape)
    score = evaluator.eval(data.y[node_ids], y_pred)[eval_metric]
    return score


def save_cpu_model():
    data, split_idx, train_idx, nlabels = precess_data()
    model = init_model(data, nlabels=nlabels, if_train=False)
    model.load_state_dict(torch.load(save_dir+'/model.pt')) #载入验证集上表现最好的模型
    model = model.to('cpu')
    print('Convert model to cpu...')
    torch.save(model.state_dict(), save_dir+'/model_cpu.pt')
    print('Done.')


def run():
    # train_all()
    inference()
    # save_cpu_model()
    # score = get_score()
    # print(score)



if __name__== "__main__":
    run()