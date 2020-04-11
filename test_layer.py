## crf layer
import torch 
import torch.nn as nn 
from torch.optim import Adam

class CRFLayer(nn.Module):
    """
    """
    def __init__(self, input_shape, lr_multiplier=1):
        super(CRFLayer, self).__init__()
        output_dim = input_shape[-1]
        # self.lr_multiplier = lr_multiplier  # 当前层学习率的放大倍数
        self.trans = nn.Parameter(torch.Tensor(output_dim, 1))
        self.trans.data.uniform_(-0.1, 0.1)


    def forward(self, x, labels):
        # predict = x.dot(self.trans)
        # predict = torch.matmul(x, self.trans)
        predict = torch.einsum("bi,ij->bj", x, self.trans)
        print("当前预测结果为：" + str(predict.detach().numpy()))
        loss = torch.sum((labels - predict)**2)
        return loss 
 

if __name__ == "__main__":
    train_x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    train_y = torch.tensor([[3], [7]])
    crf_layer = CRFLayer([2, 2])
    optim_parameters = list(crf_layer.parameters())
    optimizer = torch.optim.Adam(optim_parameters, lr=0.1, weight_decay=1e-3)

    for i in range(100):
        print("当前epoch: " + str(i))
        loss = crf_layer(train_x, train_y)
        print("loss = " + str(loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
