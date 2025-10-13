from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class EMGdataset(Dataset):

    def __init__(self, data_path, label_path):
        self.data = np.load(data_path).astype(np.float32)
        self.label = np.load(label_path).astype("int64")

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    # path = "E:\\EMG\\db5_53_u_law\\"
    path = "D:\\dataset\\"


    traindatasets = EMGdataset(path+'emg_train_db5.npy', path+'label_train_db5.npy')  # 初始化
    testdatasets = EMGdataset(path+'emg_test_db5.npy', path+'label_test_db5.npy')  # 初始化

    train_loader = DataLoader(dataset=traindatasets, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=testdatasets, batch_size=64, shuffle=True)
    #
    print(len(traindatasets))
    print(len(testdatasets))
    for data,label in train_loader:
        print(data.shape)
        print(label.dtype)
        break