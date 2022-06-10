import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def main():
    img_channels = 1
    data_root = './data/brain/brain_train'
    txt_path = 'brain/train.txt'
    slice_num = 16
    with open(txt_path) as read:
        img_name_list = [os.path.join(data_root, line.strip()) for line in read.readlines() if len(line.strip()) > 0]

    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in tqdm(img_name_list):
        for slice_index in range(slice_num):
            # 暂定切第一个维度，每个维度切16片
            slice = int(slice_index * slice_num + (256 / slice_num)/2 - 1)
            img_3d = np.load(img_name + '_data.npy')
            img = img_3d[slice, :, :]
            # plt.imshow(img, cmap='gray')
            # plt.show()
            cumulative_mean += img.mean()
            cumulative_std += img.std()

    mean = cumulative_mean / len(img_name_list) * slice_num
    std = cumulative_std / len(img_name_list) * slice_num
    print(f"mean: {mean}")
    print(f"std: {std}")


if __name__ == '__main__':
    main()
