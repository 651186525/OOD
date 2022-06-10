import os
import time

import torch
import transforms as T
import numpy as np
from PIL import Image
from my_dataset import MyDataset
import matplotlib.pyplot as plt

from src import UNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 1  # exclude background
    weights_path = "./model/sgd0.01_b128_812.007/best_model.pth"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."

    mean = (19.66,)
    std = (30.63,)
    test_dataset = MyDataset(os.getcwd(), data_type='test', transforms=T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = UNet(in_channels=1, num_classes=classes, base_c=16)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        for image, target in test_dataloader:
            # init model
            img_height, img_width = image.shape[-2:]
            init_img = torch.zeros((1, 1, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            output = model(image.to(device))
            t_end = time_synchronized()
            output = output['out']
            print("inference+NMS time: {}".format(t_end - t_start))
            pre_img = output.squeeze()
            plt.imshow(pre_img.to('cpu'), cmap='gray')
            plt.title('pre')
            plt.show()
            plt.imshow(target.squeeze(), cmap='gray')
            plt.title('GT')
            plt.show()


if __name__ == '__main__':
    main()
