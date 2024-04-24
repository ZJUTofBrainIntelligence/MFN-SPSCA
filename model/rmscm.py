import torch
import torch.nn as nn
import torch.nn.functional as F

def mish(x):
    return x*(torch.tanh(F.softplus(x)))

# XT
class XceptionTime_model(nn.Module):
    def __init__(self, in_channels, out_channels, frames, node_num):
        super().__init__()
        # 空间特征提取
        self.conv_s = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_s1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3//2)
        self.conv_s2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=5//2),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=5//2),
        )
        self.conv_s3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=9, padding=9//2*2, dilation=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=9, padding=9//2*2, dilation=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=9, padding=9//2*2, dilation=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=9, padding=9//2*2, dilation=2),
        )
        # 时间特征提取
        hidden_size = 32
        self.conv_t1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.lstm = nn.LSTM(node_num, hidden_size, bidirectional=True, batch_first=True)
        self.bn_Re = nn.Sequential(
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(inplace=True),
        )
        self.conv_t2 = nn.Conv1d(hidden_size * 2, node_num, kernel_size=1)
        self.conv_t3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3//2)


    def forward(self, x):
        N, C, T, V = x.size()     # C = 1
        # 空间
        x_s = self.conv_s(x)        # N,C,T,V
        x_s1 = self.conv_s1(x_s)    # 分支一：N,C_out,T,V
        x_s2 = self.conv_s2(x_s)    # 分支二：N,C_out,T,V
        x_s3 = self.conv_s3(x_s)    # 分支二：N,C_out,T,V
        # 时间
        x_t = self.conv_t1(x)                                   # N,C,T,V

        x_t, _ = self.lstm(x_t.view(N, T, V))                   # N,C,T,V -> N,T,V -> N,T,hid_size
        # print(x_t.shape)
        x_t = self.bn_Re(x_t.transpose(1,2).contiguous())       # N,T,hid_size -> N,hid_size,T
        # print(x_t.shape)
        x_t = self.conv_t2(x_t)                                 # N,hid_size,T -> N,V,T
        # print(x_t.shape)
        x_t = x_t.transpose(1,2).contiguous().view(N, C, T, V)  # N,V,T -> N,C,T,V
        x_t = self.conv_t3(x_t)                                 # N,C,T,V -> N,C_out,T,V

        return torch.cat([x_s1, x_s2, x_s3, x_t], 1)            # N,C,T,V -> N,4*C_out,T,V


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

#XT主函数
class Double_mstff(torch.nn.Module):
    def __init__(self, in_channels, out_channels, frames, node_num):
        super(Double_mstff, self).__init__()
        self.data_bn_imu = nn.BatchNorm1d(in_channels * 3)   # 数据预处理批次归一化
        self.data_bn_emg = nn.BatchNorm1d(in_channels * node_num)
        self.XT_imu = XceptionTime_model(in_channels, 64, frames, 3)
        self.XT_emg = XceptionTime_model(in_channels, 64, frames, node_num)

        #消融实验用到
        self.residual = nn.Sequential(      # XT结构残差
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )

        self.decode = nn.Sequential(
            Conv2d_res(256, 256),  # T, V -> T//2, V//2
            Conv2d_res(256, 128),  # T, V -> T//2, V//2
            # Conv2d_res(128, 64),  # T, V -> T//2, V//2
            Conv2d_res(128, out_channels),  # T, V -> T//2, V//2
        )

        self.SoftMax = nn.Softmax(dim=1)

    def forward(self, x):
        # 数据预处理
        # x = x.unsqueeze(1)
        # emg
        x1 = x[:, :, :, :16]
        N, C, T, V = x1.size()
        x1 = x1.transpose(2, 3).contiguous()  # N, C, V, T
        x1 = x1.view(N, C * V, T)
        x1 = self.data_bn_emg(x1)
        x1 = x1.view(N, C, V, T).transpose(2, 3).contiguous()  # N, C, T, V
        x1 = self.XT_emg(x1)
        # print(x1.shape)

        # imu
        x2 = x[:, :, :, 16:19]
        N, C, T, V = x2.size()
        x2 = x2.transpose(2, 3).contiguous()  # N, C, V, T
        x2 = x2.view(N, C * V, T)
        x2 = self.data_bn_imu(x2)
        x2 = x2.view(N, C, V, T).transpose(2, 3).contiguous()  # N, C, T, V
        x2 = self.XT_imu(x2)
        # print(x2.shape)

        y = torch.cat([x1, x2], 3) #50，52，40，19
        return y
        #消融实验处理
        # res = self.residual(x)
        # y = mish(y + res)
        # #解码
        # sum = self.decode(y)  # N,64
        # # T,V纬度上池化
        # sum = F.avg_pool2d(sum, sum.size()[2:])
        # # 去掉多余纬度
        # sum = torch.squeeze(sum)
        # return self.SoftMax(sum)


if __name__ == '__main__':
    x1 = torch.rand(50, 1, 40, 16)
    mo = XceptionTime_model(1, 64, 40, 16)
    print(mo(x1).shape)