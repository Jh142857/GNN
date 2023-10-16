## 生成 main.py 时请勾选此 cell
from itertools import count
from utils import DGraphFin
from utils.evaluator import Evaluator
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import Data
import numpy as np
import os
from torch_geometric.nn import GATConv, GCNConv


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


flag = True
y_pred_all = torch.empty((0,0))

def predict(data,node_id):
    """
    加载模型和模型预测
    :param node_id: int, 需要进行预测节点的下标
    :return: tensor, 类0以及类1的概率, torch.size[1,2]
    """
    global y_pred_all, flag
    device = 0
    device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    nlabels = 2
    gat_parameters = {
        'lr': 0.01
        , 'num_layers': 2
        , 'hidden_channels': 16
        , 'dropout': 0.0
        , 'batchnorm': False
        , 'weight_decay': 5e-7
        , 'resume_epoch': 800
    }
    para_dict = gat_parameters
    model_para = gat_parameters.copy()
    model_para.pop('lr')
    model_para.pop('weight_decay')
    
    # 这里可以加载你的模型
    model = GAT_NET(data.x.size(-1), hidden=gat_parameters['hidden_channels'], classes=nlabels, heads=2).cpu()
    model.load_state_dict(torch.load('./results/gat/model-2layers-16hidden-1000epochs.pt', map_location=torch.device('cpu')))
    # 模型预测时，测试数据已经进行了归一化处理
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    if(flag):
        with torch.no_grad():
            model.eval()
            out = model(data.x, data.adj_t)
            y_pred_all = out.exp()  # (N,num_classes)
        flag = False
    y_pred = y_pred_all[node_id]    
    
    return y_pred