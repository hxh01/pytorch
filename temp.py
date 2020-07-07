import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

x = range(1, 21, 1)
y = range(100, 2100, 100)


class XYDataSet(Dataset):
    def __init__(self, x, y):
        self.x_list = x
        self.y_list = y
        assert len(self.x_list) == len(self.y_list)

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, index):
        x_one = self.x_list[index]
        y_one = self.y_list[index]
        return (x_one, y_one)


# 第一步：构造dataset
dataset = XYDataSet(x, y)
# 第二步：构造dataloader
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

# 第三步：对dataloader进行迭代
for epoch in range(2):  # 只查看两个epoch
    for x_train, y_train in dataloader:
        print(x_train)
        print(y_train)
        print("-----------------------------------")
