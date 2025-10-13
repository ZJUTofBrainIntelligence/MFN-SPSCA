from datetime import datetime

from juece import *
from data_load import EMGdataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from sklearn.manifold import TSNE
import math
import os


def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.GRU) or isinstance(m, nn.LSTM) or isinstance(m, nn.RNN):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.normal_(param, mean=0, std=0.02)


def confusion_matrix(preds, labels, conf_matrix):  # 混淆矩阵填充函数 行为真实值，列为预测值
    preds = torch.argmax(preds, dim=1)
    for t, p in zip(labels, preds):
        conf_matrix[t, p] += 1
    return conf_matrix


def tSNE_plot(data, true_label, dim, prep, labels, size, save_path=None):
    # data是高维数据，形状为（true_label.shape[0], class）,来自data = model(x_test)
    # perp的取值根据个人经验取值，一般样本数在0-999时prep取值在5-20，在1000-9999时取值在20-50，>10000时取值为50-100  模糊度
    # dim是维数，可以选择tSNE绘制为2D图还是3D图
    #
    # labels必须按照标签0-n排列，类型为字符串
    # size 是圆点的大小 可自行调节
    selected_labels = np.array([5, 34, 51])
    data = data.cpu().detach().numpy()
    true_label = np.array(true_label, dtype=int)
    indices = np.where(
        (true_label == selected_labels[0]) | (true_label == selected_labels[1]) | (true_label == selected_labels[2]))
    data = data[indices]
    true_label = true_label[indices]
    tsne = TSNE(n_components=dim, perplexity=prep, random_state=42)
    X_tsne = tsne.fit_transform(data)
    if dim == 2:
        fig, ax = plt.subplots(figsize=(6, 8))
        for i, label in enumerate(np.unique(true_label)):
            indices = true_label == label
            ax.scatter(X_tsne[indices, 0], X_tsne[indices, 1], s=size, label=labels[i])
    elif dim == 3:
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i, label in enumerate(np.unique(true_label)):
            indices = true_label == label
            ax.scatter(X_tsne[indices, 0], X_tsne[indices, 1], X_tsne[indices, 2], s=5, label=labels[i])
    else:
        raise ValueError(f"Unsupported dimension {dim}. Only 2D and 3D visualizations are supported.")
    ax.legend()
    # ax.set_title('tSNE')
    if not save_path == None:
        plt.savefig('{}.png'.format(save_path))


class Processor():
    def __init__(self, result_path='../result/', file_name='EMG_reco',
                 graph_args={"split_type": 'three partitions', "TH": 0.1}):
        assert isinstance(result_path, str) and isinstance(file_name, str)
        if not result_path.endswith('\\') and not result_path.endswith('/'):
            result_path += '/'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_feature = 1  # 输入特征
        self.in_frames = 40  # 输入帧数
        self.class_num = 53  # 输出分类
        self.node_num = 16  # 节点数
        self.graph_args = graph_args  # {"split_type":'spatial',"TH":0.5}   # 邻接矩阵分区策略
        # self.model_choose = model_choose # 加载模型选择
        self.imp = True  # stgcn的可学习边权重机制
        self.epoch = 30  # 训练的轮数
        self.train_batch_size = 64  # 训练集单批次尺寸
        self.test_batch_size = 64  # 测试集单批次尺寸
        self.dropout = 0
        self.test_pro = 1  # 测试集数量过大利用多少比例的数据进行测试
        self.opti_choose = 'SGD'  # 优化器选择
        self.nesterov = True  # SGD是否使用nesterov（SGD优化器有效）
        self.weight_decay = 0.0001  # 权重衰减值（SGD优化器有效） stgcn代码中为0.0001
        self.base_lr = 0.01  # 基础学习率
        self.epoch_lr = 5  # 学习率调整轮次，支持例每n轮调整，如10，或者指定超过多少轮后调整如，[10,20,30]
        self.gamma_lr = 0.5  # 学习率调整因子
        self.data_path = 'D:\\pycharm file\\xt_cwt_gcn_db5\\dataset\\'
        self.data_name = ['data_train_db5_53.npy', 'label_train_db5_53.npy', 'data_test_db5_53.npy',
                          'label_test_db5_53.npy']
        self.file_name = file_name  # 保存文件名前缀
        self.result_path = result_path  # 结果、模型的保存路径
        self.old_model = None  # 加载模型的路径（选填）
        self.old_lr = 0  # 加载模型后开始的学习率（选填）
        self.old_epoch = 0  # 加载模型的轮次（选填）
        self.cur_epoch = 1  # 当前轮次
        # 设置训练网络的一些参数
        self.train_reco_loss = []  # 每轮训练的识别损失
        self.test_reco_loss = []  # 每轮测试的识别损失
        self.train_accuracy = []  # 每轮训练的准确率
        self.test_accuracy = []  # 每轮测试的准确率
        self.train_precision = []  # 每轮训练的查准率
        self.test_precision = []  # 每轮测试的查准率
        self.train_recall = []  # 每轮训练的查全率
        self.test_recall = []  # 每轮测试的查全率
        self.train_f1 = []  # 每轮训练的F1
        self.test_f1 = []  # 每轮测试的F1
        self.loss_fig, self.loss_ax = plt.subplots()  # 生成loss画布
        self.accu_fig, self.accu_ax = plt.subplots()  # 生成准确率画布
        self.conf_matrix_fig, self.conf_matrix_ax = plt.subplots()  # 混淆矩阵的画布

    def start(self):
        self.load_data()  # 加载数据
        self.load_model()  # 加载模型、优化器、损失

        # 创建csv文件，记录loss和accuracy
        result_csv = pd.DataFrame(columns=['total time', 'epoch time',
                                           'train reco loss', 'test reco loss',
                                           'train accuracy', 'test accuracy',
                                           'train precision', 'test precision',
                                           'train recall', 'test recall',
                                           'train f1', 'test f1'])  # 列名
        result_csv.to_csv(self.result_path + self.file_name + "_result_data.csv", index=False)
        # pd.read_csv('F:\\Documents\\train_acc.csv')
        # label = np.array(range(53))
        label = ['gesture 5', 'gesture 34', 'gesture 51']
        save_path1 = './result_tsne'
        self.start_time = datetime.now()
        for self.cur_epoch in range(self.cur_epoch, self.epoch + 1):
            cur_lr = self.optimizer.param_groups[0]['lr']
            print("-------第 {} 轮训练开始，当前学习率{}-------".format(self.cur_epoch, cur_lr))
            self.last_time = datetime.now()
            self.train()
            self.test()
            self.scheduler.step()  # 学习率调整
            self.total_time = datetime.now() - self.start_time
            self.epoch_time = datetime.now() - self.last_time
            self.save_result()
            self.draw()
            self.save_model()
        tSNE_plot(self.all_outputs, self.all_true, 2, 10, label, 25, save_path1)
        print("-------训练完成-------")

    def train(self):
        self.model.train()  # model变为训练模式
        total_reco_loss = 0
        total_accuracy = 0
        total_label = np.empty(0)  # 所有数据的标签
        total_reco = np.empty(0)  # 所有的预测结果
        train_bar = tqdm(self.train_dataloader)

        for i, (data, label) in enumerate(train_bar):
            # date获取格式为[N, T, V] label获取格式为[N]
            data = data.to(self.device)  # 移至GPU
            label = label.to(self.device)

            # db1修正
            # label -= 1

            self.optimizer.zero_grad()  # 梯度置零
            reco_out = self.model(data)
            reco_loss = self.loss_fn_reco(torch.log(reco_out), label)  # 交叉熵=softmax+log+nullloss

            # 安全loss 防log(0)
            if reco_loss == float("inf"):
                NEAR_0 = 1e-45
                reco_loss = self.loss_fn_reco(torch.log(reco_out + NEAR_0), label)

            # 优化器优化模型
            reco_loss.backward()
            self.optimizer.step()

            total_reco_loss += reco_loss.item()
            accuracy = int((reco_out.argmax(1) == label).sum()) / len(label)
            total_accuracy += accuracy

            label_array = label.clone().detach().cpu().numpy()
            reco_array = reco_out.argmax(1).clone().detach().cpu().numpy()
            total_label = np.concatenate((total_label, label_array), axis=0)
            total_reco = np.concatenate((total_reco, reco_array), axis=0)

            train_bar.desc = "train epoch[{}/{}] loss:{:.5f} accu:{:.2f}".format(self.cur_epoch, self.epoch, reco_loss,
                                                                                 accuracy)

        self.train_reco_loss.append(float(total_reco_loss) / self.train_times)
        self.train_accuracy.append(total_accuracy / self.train_times)
        self.train_precision.append(precision_score(total_label, total_reco, average='macro'))
        self.train_recall.append(recall_score(total_label, total_reco, average='macro'))
        self.train_f1.append(f1_score(total_label, total_reco, average='macro'))

    def test(self):
        self.model.eval()
        all_outputs = []
        all_true = []
        total_reco_loss = 0
        total_accuracy = 0
        total_label = np.empty(0)  # 所有数据的标签
        total_reco = np.empty(0)  # 所有的预测结果
        self.conf_matrix = torch.zeros(self.class_num, self.class_num)  # 空混淆矩阵 行为真实值，列为预测值
        # 用于记录最后一轮次的测试集的标签与各个分类置信度
        label_confid_csv = pd.DataFrame(columns=['true label'] + ['confid {}'.format(i) for i in range(self.class_num)])
        label_confid_csv.to_csv(self.result_path + self.file_name + "_label_confid.csv", index=False)

        test_bar = tqdm(self.test_dataloader)

        with torch.no_grad():
            for i, (data, label) in enumerate(test_bar):
                if i >= self.test_times:
                    break
                # date获取格式为[N, T, V] label获取格式为[N]
                data = data.to(self.device)  # 移至GPU
                label = label.to(self.device)
                all_true.extend(label.cpu().tolist())
                # db1修正
                # label -= 1

                reco_out = self.model(data)
                all_outputs.append(reco_out)

                reco_loss = self.loss_fn_reco(torch.log(reco_out), label)  # 交叉熵=softmax+log+nullloss

                # 安全loss 防log(0)
                if reco_loss == float("inf"):
                    NEAR_0 = 1e-45
                    reco_loss = self.loss_fn_reco(torch.log(reco_out + NEAR_0), label)

                total_reco_loss += reco_loss.item()
                accuracy = int((reco_out.argmax(1) == label).sum()) / len(label)
                total_accuracy += accuracy

                label_array = label.clone().detach().cpu().numpy()
                reco_array = reco_out.argmax(1).clone().detach().cpu().numpy()
                total_label = np.concatenate((total_label, label_array), axis=0)
                total_reco = np.concatenate((total_reco, reco_array), axis=0)
                self.conf_matrix = confusion_matrix(reco_out, label, self.conf_matrix)

                label_conf = pd.DataFrame(torch.cat([label.unsqueeze(1), reco_out], dim=1).tolist())
                label_conf.to_csv(self.result_path + self.file_name + "_label_confid.csv", mode='a', header=False,
                                  index=False)  # mode设为a,就可以向csv文件追加数据了

                test_bar.desc = "test epoch[{}/{}] loss:{:.5f} accu:{:.2f}".format(self.cur_epoch, self.epoch,
                                                                                   reco_loss, accuracy)
                # if i == 100:
                #     break
            self.all_true = all_true
            self.all_outputs = torch.cat(all_outputs, dim=0)
            self.test_reco_loss.append(float(total_reco_loss) / self.test_times)
            self.test_accuracy.append(total_accuracy / self.test_times)
            self.test_precision.append(precision_score(total_label, total_reco, average='macro'))
            self.test_recall.append(recall_score(total_label, total_reco, average='macro'))
            self.test_f1.append(f1_score(total_label, total_reco, average='macro'))

    def draw(self, show_flag=False):
        self.loss_ax.plot(range(len(self.train_reco_loss)), self.train_reco_loss, label="train loss", color='m')
        self.loss_ax.plot(range(len(self.test_reco_loss)), self.test_reco_loss, label="test loss", color='y')
        self.loss_ax.set_title("model loss")
        self.loss_ax.set_xlabel("epoch")
        self.loss_ax.set_ylabel("loss")
        self.loss_ax.set_ylim([0, max(self.train_reco_loss + self.test_reco_loss)])
        self.loss_ax.legend(loc='upper right')

        self.accu_ax.plot(range(len(self.train_accuracy)), self.train_accuracy, label="train accuracy", color='r')
        self.accu_ax.plot(range(len(self.test_accuracy)), self.test_accuracy, label="test accuracy", color='g')
        self.accu_ax.set_title("model accuracy")
        self.accu_ax.set_xlabel("epoch")
        self.accu_ax.set_ylabel("accuracy")
        self.accu_ax.set_ylim([0, 1])
        self.accu_ax.legend(loc='upper left')

        self.conf_matrix_ax.imshow(self.conf_matrix, cmap=plt.cm.Blues)

        self.loss_fig.savefig(self.result_path + 'model loss.png')
        self.accu_fig.savefig(self.result_path + 'model accuracy.png')
        self.conf_matrix_fig.savefig(self.result_path + 'model conf matrix.png')
        print("-------图像保存成功-------")
        if show_flag:
            plt.show()

        plt.sca(self.loss_ax)  # 保存后就删除axes
        plt.cla()
        plt.sca(self.accu_ax)
        plt.cla()
        plt.sca(self.conf_matrix_ax)
        plt.cla()

    def save_model(self):
        torch.save(self.model.state_dict(), self.result_path + self.file_name + "_epoch{}.pt".format(self.cur_epoch))
        # {'optimizer_dict': self.optimizer.state_dict(), 'model_dict': self.model.state_dict()}

        # 删除前一个模型
        if self.cur_epoch > 1:
            os.remove(self.result_path + self.file_name + "_epoch{}.pt".format(self.cur_epoch - 1))
        print("-------模型保存成功-------")

    def load_model(self):

        self.model = trans_fusion(self.in_feature, self.class_num, self.graph_args, frames=self.in_frames,
                                  node_num=self.node_num, edge_importance_weighting=self.imp, dropout=self.dropout).to(
            self.device)

        self.loss_fn_reco = torch.nn.NLLLoss().to(self.device)

        if self.old_model:  # 加载指定模型权重
            self.model.load_state_dict(torch.load(self.old_model))
            self.cur_epoch = self.old_epoch + 1
        else:
            self.model.apply(weights_init)
            self.old_lr = self.base_lr
            self.old_epoch = 0

        if self.opti_choose == 'SGD':  # 选择优化器
            self.optimizer = optim.SGD([{'params': self.model.parameters(), 'initial_lr': self.base_lr}],
                                       lr=self.old_lr,
                                       momentum=0.9, nesterov=self.nesterov, weight_decay=self.weight_decay)
        elif self.opti_choose == 'Adam':
            self.optimizer = optim.Adam([{'params': self.model.parameters(), 'initial_lr': self.base_lr}],
                                        lr=self.old_lr, weight_decay=self.weight_decay)
        else:
            raise ValueError("Optimizer selection error!")

        if isinstance(self.epoch_lr, int):  # 设置优化器学习率自动调整 last_epoch从0开计
            self.scheduler = StepLR(self.optimizer, self.epoch_lr, gamma=self.gamma_lr, last_epoch=self.old_epoch - 1)
        elif isinstance(self.epoch_lr, list):
            self.scheduler = MultiStepLR(self.optimizer, self.epoch_lr, gamma=self.gamma_lr,
                                         last_epoch=self.old_epoch - 1)
        else:
            raise ValueError("epoch_lr setting error!")

        print("模型加载完成")

    def load_data(self):
        # 获取数据集
        self.train_dataset = EMGdataset(self.data_path + self.data_name[0], self.data_path + self.data_name[1])
        self.test_dataset = EMGdataset(self.data_path + self.data_name[2], self.data_path + self.data_name[3])

        # length 长度
        self.train_data_size = len(self.train_dataset)
        self.test_data_size = len(self.test_dataset)
        self.test_effe_size = int(self.test_data_size * self.test_pro)

        self.train_times = self.train_data_size // self.train_batch_size  # 每轮训练次数
        self.test_times = self.test_effe_size // self.test_batch_size  # 每轮测试次数
        print("训练数据集的长度为：{}".format(self.train_data_size))
        print("测试数据集的长度为：{}, 每轮测试量为{}".format(self.test_data_size, self.test_effe_size))

        # 利用 DataLoader 来加载数据集  加载结果类型为tensor 舍弃不足一批次的数据量的尾部文件
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True,
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=True,
                                          drop_last=True)
        print("训练总轮数{}, 每轮训练次数{}, 每轮测试次数{}"
              .format(self.epoch, self.train_times, self.test_times))

    def save_result(self):
        cur_index = self.cur_epoch - 1
        result_data = pd.DataFrame([[self.total_time, self.epoch_time,
                                     self.train_reco_loss[cur_index], self.test_reco_loss[cur_index],
                                     self.train_accuracy[cur_index], self.test_accuracy[cur_index],
                                     self.train_precision[cur_index], self.test_precision[cur_index],
                                     self.train_recall[cur_index], self.test_recall[cur_index],
                                     self.train_f1[cur_index], self.test_f1[cur_index]]])  # 注意要转为二维
        result_data.to_csv(self.result_path + self.file_name + "_result_data.csv", mode='a', header=False,
                           index=False)  # mode设为a,就可以向csv文件追加数据了
        print("已经耗时：{}, 本轮耗时：{}, 训练集识别loss：{}, 测试集识别loss：{}, 训练集准确率：{}, 测试集准确率：{}"
              .format(self.total_time, self.epoch_time, self.train_reco_loss[cur_index], self.test_reco_loss[cur_index],
                      self.train_accuracy[cur_index], self.test_accuracy[cur_index]))


if __name__ == '__main__':
    p = Processor(result_path='./result', graph_args={"split_type": 'three partitions', "TH": 0.1})
    p.start()