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
from torch_geometric.nn import GraphConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, feature, hidden, classes, layers=2) -> None:
        super().__init__()
        self.sage1 = GraphConv(feature, hidden)
        self.sage_middle = GraphConv(hidden, hidden)
        self.sage2 = GraphConv(hidden, classes)
        self.hidden_layers = layers - 2
    
    def forward(self, x, adj_t):
        x = self.sage1(x, adj_t)
        x = F.relu(x)
        # x = F.dropout(x)
        for _ in range(self.hidden_layers):
            x = self.sage_middle(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x)
        x = self.sage2(x, adj_t)

        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        self.sage1.reset_parameters()
        self.sage2.reset_parameters()


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
    sage_parameters = {
        'lr': 0.01
        , 'num_layers': 2
        , 'hidden_channels': 16
        , 'dropout': 0.0
        , 'batchnorm': False
        , 'weight_decay': 5e-7
        , 'resume_epoch': 800
    }
    para_dict = sage_parameters
    model_para = sage_parameters.copy()
    model_para.pop('lr')
    model_para.pop('weight_decay')
    
    # 这里可以加载你的模型
    model = GraphSAGE(data.x.size(-1), hidden=sage_parameters['hidden_channels'], classes=nlabels, layers=sage_parameters['num_layers']).cpu()
    model.load_state_dict(torch.load('./results/graph/model-2layers-16hidden-1500epochs.pt', map_location=torch.device('cpu')))
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