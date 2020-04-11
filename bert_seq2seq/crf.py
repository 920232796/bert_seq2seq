## crf layer
import torch 
import torch.nn as nn 
import torch.nn.functional as F

class CRFLayer(nn.Module):
    """
    """
    def __init__(self, input_shape, lr_multiplier=1):
        super(CRFLayer, self).__init__()
        output_dim = input_shape[-1]
        self.lr_multiplier = lr_multiplier  # 当前层学习率的放大倍数
        self.trans = nn.Parameter(torch.Tensor(output_dim, output_dim))
        self.trans.data.uniform_(-0.1, 0.1)


    def forward(self, x, labels):
        pass
    
    def target_score(self, y_true, y_pred):
        """
        计算状态标签得分 + 转移标签得分
        y_true: (batch, seq_len, out_dim)
        y_pred: (batch, seq_len, out_dim)
        """
        point_score = torch.einsum("bni,bni->b", y_pred, y_true)
        trans_score = torch.einsum("bni,ij,bnj->b", y_true[:, :-1], self.trans, y_true[:, 1: ])

        return point_score + trans_score
    
    def log_norm_step(self, inputs, states):
        """
        计算归一化因子Z(X)
        """

    def forward(self, y_pred, y_true):
        """
        """
        y_true_onehot = F.one_hot(y_true, self.trans.shape[0])

 

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
    
