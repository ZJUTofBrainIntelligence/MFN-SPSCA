import pywt
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from stgcm_sps import *

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

class CWT_emg(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.bn2 = torch.nn.BatchNorm2d(128)

        self.Z3 = nn.Conv2d(128, 128, kernel_size=[1, 38])
        self.bn3 = torch.nn.BatchNorm2d(128)

    def forward(self, x0):
        # CWT 转换
        cwt_tensor = []
        for i in range(x0.shape[0]):
            cwt_ch = []
            for j in range(x0.shape[3]):
                cwt = pywt.cwt(x0[i, 0, :, j].cpu().numpy(), np.arange(1, 41), 'morl')[0]
                cwt_ch.append(cwt)
            cwt_tensor.append(cwt_ch)
        x = torch.tensor(np.array(cwt_tensor), dtype=torch.float32).to('cuda') #50,8,40,40
        # print(x.shape)
        x = self.conv(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # 蒙特卡洛dropout
        # x = self.mcdropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.Z3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x

class CWT_IMU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.bn2 = torch.nn.BatchNorm2d(128)

        self.Z3 = nn.Conv2d(128, 128, kernel_size=[1, 38])
        self.bn3 = torch.nn.BatchNorm2d(128)

    def forward(self, x0):
        # CWT 转换
        cwt_tensor = []
        for i in range(x0.shape[0]):
            cwt_ch = []
            for j in range(x0.shape[3]):
                cwt = pywt.cwt(x0[i, 0, :, j].cpu().numpy(), np.arange(1, 41), 'morl')[0]
                cwt_ch.append(cwt)
            cwt_tensor.append(cwt_ch)
        x = torch.tensor(np.array(cwt_tensor), dtype=torch.float32).to('cuda') #50,3,32,32
        # print(x.shape)
        x = self.conv(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.Z3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x

class scnn_gcn(torch.nn.Module):
    def __init__(self, in_channels, class_num):
        super(scnn_gcn, self).__init__()
        self.emg = Model(1, 53)
        self.imu = CWT_IMU()
        self.emgcwt = CWT_emg()


        self.residual = nn.Sequential(  # 结构残差
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )

        # 解码
        self.decode = nn.Sequential(
            Conv2d_res(256, 256),  # T, V -> T//2, V//2
            Conv2d_res(256, 128),  # T, V -> T//2, V//2
            Conv2d_res(128, 64),  # T, V -> T//2, V//2
            # Conv2d_res(128, class_num),  # T, V -> T//2, V//2
        )

        self.SoftMax = nn.Softmax(dim=1)


    def forward(self, x):
        #emg-gcn
        # x = x.unsqueeze(1)
        x11 = x[:, :, :, :16]
        # print(x1.shape)
        x1 = self.emg(x11) #50,128,1,1
        # print(x1.shape)
        x3 = self.emgcwt(x11) #50,128,32,3
        # print(x3.shape)
        #imu-cwt
        x2 = x[:, :, :, 16:19]
        # print(x2.shape)
        x2 = self.imu(x2) #50,128,40,3
        # print(x2.shape)
        #特征融合

        sum0 = torch.cat([x2, x3], 1)
        sum = torch.cat([x1, sum0], 3)
        # res = self.residual(x)
        # y = mish(sum + res)

        return sum


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('#### Test Model ###')

    x = torch.rand(50, 1, 40, 19).to(device)
    model = scnn_gcn(1, 53).to(device)

    y = model(x)

    print(y.shape)