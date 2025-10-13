import torch
import torch.nn as nn
import torch.nn.functional as F
from mstff_for import Double_mstff
from cwt_cm import scnn_gcn
from crosstransformer2 import Cross2
from thop import profile
from thop import clever_format
from torchinfo import summary

def mish(x):
    return x*(torch.tanh(F.softplus(x)))

class Conv2d_res(nn.Module):

    def __init__(self, in_channels, out_channels, residual=True, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, stride=2),
            nn.BatchNorm2d(out_channels),
        )
        if not residual:
            self.residual = lambda x: 0
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.conv(x)
        return mish(x + res)

class trans_fusion(nn.Module):
    def __init__(self, in_feature, class_num, graph_args={}, frames=40, node_num=16, edge_importance_weighting=True, dropout=0):
        super(trans_fusion, self).__init__()
        self.dmst = Double_mstff(in_feature, class_num, frames=frames, node_num=node_num)
        self.dsc_gcn = scnn_gcn(in_feature, class_num)
        self.cross_trans2 = Cross2(19, 256)

        self.residual = nn.Sequential(  # 结构残差
            nn.Conv2d(in_feature, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )

        # 解码
        self.decode = nn.Sequential(
            Conv2d_res(256, 256),  # T, V -> T//2, V//2
            Conv2d_res(256, 128),  # T, V -> T//2, V//2
            Conv2d_res(128, class_num),  # T, V -> T//2, V//2
        )

        self.SoftMax = nn.Softmax(dim=1)

    def forward(self, xinput):
        data = xinput.unsqueeze(1)  # 数据预处理[N, T, V]->[N, C, T, V]
        x1 = self.dmst(data)  # 50,256,40,19
        x2 = self.dsc_gcn(data)  # 50,256,40,19

        x1 = x1.permute(0, 2, 3, 1) #50,40,19,256
        N1, T1, C1, V1 = x1.size()
        x1 = x1.reshape(N1*T1, C1, V1)
        # print(x1.shape)

        x2 = x2.permute(0, 2, 3, 1)#50,40,19,256
        N2, T2, C2, V2 = x2.size()
        x2 = x2.reshape(N2 * T2, C2, V2)
        # print(x2.shape)

        y = self.cross_trans2(x2, x1)
        NT, C, V = y.size()
        y = y.reshape(-1, 40, C, V)
        y = y.permute(0, 3, 1, 2)  # 50,256,40,19

        res = self.residual(data)
        sum = mish(y + res)

        # 解码
        sum = self.decode(sum)  # N,class_num,5,2
        # T,V纬度上池化
        sum = F.avg_pool2d(sum, sum.size()[2:])
        # 去掉多余纬度
        sum = torch.squeeze(sum)
        return self.SoftMax(sum)

        
if __name__ == '__main__':
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # device = torch.device("cuda:0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x1 = torch.rand(50, 40, 19).to(device)
    mo = trans_fusion(1, 53).to(device)
    print(mo(x1).shape)

    flops, params = profile(mo.to(device), inputs=(x1,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

    # # # Define input
    x1 = torch.rand(50, 40, 19)
    # Create model instance
    model = trans_fusion(1, 53)

    # Move model and input to the available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x1 = x1.to(device)

    # Get model summary
    print(summary(model, input_data=x1))


