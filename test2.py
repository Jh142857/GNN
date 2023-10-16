from utils import DGraphFin
import torch_geometric.transforms as T
# import torch
from torch_geometric.data import Data
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv

device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

path='./datasets/632d74d4e2843a53167ee9a1-momodel/' #数据保存路径
save_dir='./results/' #模型保存路径
dataset_name='DGraph'
dataset = DGraphFin(root=path, name=dataset_name, transform=T.ToSparseTensor())
data = dataset[0]

data.x = data.x.cuda()
# print(data.x.shape)
data.y = data.y.squeeze(1).cuda()
data.adj_t = data.adj_t.cuda()

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
    

model = GAT_NET(dataset.num_features, 16, dataset.num_classes, heads=2).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

x = model(data.x, data.adj_t)
print(x.shape)
def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(1, 201):
    loss = train(data)
    print(loss, epoch)