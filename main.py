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