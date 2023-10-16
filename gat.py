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
from torch_geometric.nn import GATConv, GCNConv

import time
# 设置gpu设备
device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
print(f'设备：{device}')

# 模型设置
model_name = 'gat'

# 参数
gat_parameters = {
    'lr': 0.01
    , 'num_layers': 2
    , 'hidden_channels': 16
    , 'dropout': 0.0
    , 'batchnorm': False
    , 'weight_decay': 5e-7
    , 'resume_epoch': 800
}

epochs = 1000
log_steps =10 # log记录周期
para_dict = gat_parameters
# 评价指标
eval_metric = 'auc'  #使用AUC衡量指标
evaluator = Evaluator(eval_metric)

# 路径准备
path='./datasets/632d74d4e2843a53167ee9a1-momodel/' #数据保存路径
save_dir='./results/'+model_name #模型保存路径
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
dataset_name='DGraph'
resume_path = f'/model-{gat_parameters["num_layers"]}layers-{gat_parameters["hidden_channels"]}hidden-{gat_parameters["resume_epoch"]}epochs.pt'
final_path = f'/model-{gat_parameters["num_layers"]}layers-{gat_parameters["hidden_channels"]}hidden-{epochs}epochs.pt'
y_pred_all = torch.empty((0,0))
flag = True

class GCN_NET(torch.nn.Module):

    def __init__(self, features, hidden, classes):
        super(GCN_NET, self).__init__()
        self.conv1 = GCNConv(features, hidden)  # shape（输入的节点特征维度 * 中间隐藏层的维度）
        self.conv2 = GCNConv(hidden, classes)  # shaape（中间隐藏层的维度 * 节点类别）

    def forward(self, x, adj_t):
        # 传入卷积层
        x = self.conv1(x, adj_t)
        x = F.relu(x)  # 激活函数
        # x = F.dropout(x, training=self.training)  # dropout层，防止过拟合
        x = self.conv2(x, adj_t)  # 第二层卷积层
        # 将经过两层卷积得到的特征输入log_softmax函数得到概率分布
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


class GAT_NET(torch.nn.Module):
    def __init__(self, features, hidden, classes, heads=4):
        super(GAT_NET, self).__init__()
        self.gat1 = GATConv(features, hidden, heads)  # 定义GAT层，使用多头注意力机制
        self.gat2 = GATConv(hidden*heads, classes)  # 因为多头注意力是将向量拼接，所以维度乘以头数。

    def forward(self, x, adj_t):
        x = self.gat1(x, adj_t)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.gat2(x, adj_t)

        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        self.gat1.reset_parameters()
        self.gat2.reset_parameters()


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
    # 降维，用于解决之后的交叉熵问题
    if data.y.dim() == 2:
        data.y = data.y.squeeze(1)

    # 转化成GPU类型张量
    data.x = data.x.cuda()
    data.y = data.y.cuda()
    data.adj_t = data.adj_t.cuda()

    split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}  #划分训练集，验证集

    train_idx = split_idx['train']
    return data, split_idx, train_idx, nlabels


def init_model(data, nlabels=2, if_train=True):
    # 初始化模型
    # model = MLP(in_channels=data.x.size(-1), out_channels=nlabels, **model_para).to(device)
    # model = GAT_NET(data.x.size(-1), hidden=16, classes=nlabels).to(device)
    model = GAT_NET(data.x.size(-1), hidden=gat_parameters['hidden_channels'], classes=nlabels, heads=2).cuda()
    if if_train:
        print(f'模型:{model_name}')
        print(f'模型参数总量：{sum(p.numel() for p in model.parameters())}')  #模型总参数量
    return model


# 一个epoch训练
def train(model, data, train_idx, optimizer):
     # data.y is labels of shape (N, ) 
    model.train()

    optimizer.zero_grad()
    
    out = model(data.x, data.adj_t)

    loss = F.nll_loss(out[train_idx], data.y[train_idx])
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
            
            out = model(data.x, data.adj_t)
            y_pred = out.exp()  # (N,num_classes)
            
            # 负对数似然损失
            losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()
            eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])[eval_metric]

    return eval_results, losses, y_pred



def train_multiple(model, data, train_idx, split_idx, save_dir, resume_epoch=0):
    model.reset_parameters()
    if resume_epoch and resume_epoch < epochs:
        print(f"从{resume_epoch}个epoch恢复训练")
        resume_pt = save_dir + resume_path
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
            
            torch.save(model.state_dict(), save_dir+final_path) #将表现最好的模型保存

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
    global y_pred_all, flag
    if flag:           
        with torch.no_grad():
            model.eval()
            # print(data.adj_t)
            out = model(data.x, data.adj_t)
            y_pred_all = out.exp()  # (N,num_classes)
        flag = False
    y_pred = y_pred_all[node_id]    
    return y_pred


def train_all(resume_epoch=0):
    data, split_idx, train_idx, nlabels = precess_data()
    model = init_model(data, nlabels=nlabels)
    train_multiple(model, data, train_idx, split_idx, save_dir, resume_epoch=resume_epoch)


def inference():
    data, split_idx, train_idx, nlabels = precess_data()
    model = init_model(data, nlabels=nlabels, if_train=False)
    model.load_state_dict(torch.load(save_dir + final_path)) #载入验证集上表现最好的模型

    start_time = time.time()
    dic={0:"正常用户",1:"欺诈用户"}
    node_idx = 0
    y_pred = predict(model, data, node_idx)
    print(y_pred)
    print(f'节点 {node_idx} 预测对应的标签为:{torch.argmax(y_pred)}, 为{dic[torch.argmax(y_pred).item()]}。')

    node_idx = 1
    y_pred = predict(model, data, node_idx)
    print(y_pred)
    print(f'节点 {node_idx} 预测对应的标签为:{torch.argmax(y_pred)}, 为{dic[torch.argmax(y_pred).item()]}。')
    end_time = time.time()

    node_idx = 2
    y_pred = predict(model, data, node_idx)
    print(y_pred)
    print(f'节点 {node_idx} 预测对应的标签为:{torch.argmax(y_pred)}, 为{dic[torch.argmax(y_pred).item()]}。')
    end_time = time.time()
    node_idx = 3
    y_pred = predict(model, data, node_idx)
    print(y_pred)
    print(f'节点 {node_idx} 预测对应的标签为:{torch.argmax(y_pred)}, 为{dic[torch.argmax(y_pred).item()]}。')
    end_time = time.time()
    print(end_time - start_time)


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
    # train_all(gat_parameters['resume_epoch'])
    inference()
    # save_cpu_model()
    # score = get_score()
    # print(score)



if __name__== "__main__":
    run()