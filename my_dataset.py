import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image, to_tensor


class MyDataset(Dataset):
    def __init__(self, root: str, data_type: str, transforms=None, slice_num: int = 16):
        super(MyDataset, self).__init__()
        assert data_type in ['train', 'val', 'test'], "data type must in ['train', 'val', 'test']"
        data_root = os.path.join(root, 'data', "brain", 'brain_train' if data_type in ['train', 'val'] else 'toy')
        txt_path = os.path.join(root, 'data', 'brain', data_type + '.txt')
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        assert os.path.exists(txt_path), 'not found {} file'.format(data_type + '.txt')
        self.transforms = transforms
        self.slice_num = slice_num
        # 3D 数据路径
        with open(txt_path) as read:
            self.img_name_list = [os.path.join(data_root, line.strip()) for line in read.readlines() if
                                  len(line.strip()) > 0]

        # check file
        assert len(self.img_name_list) > 0, 'in "{}" file does not find any information'.format(data_type + '.txt')
        for img_path in self.img_name_list:
            assert os.path.exists(img_path + '_data.npy'), 'not found "{}" file'.format(img_path)

    def __getitem__(self, idx):
        # 2D切片：将brain共256个切片， 取出16个切片，第一个切片为索引7（第八个），第二个为23，依次类推  32
        slice_interval = 256 / self.slice_num
        img_index = idx // self.slice_num
        slice_index = int((idx % self.slice_num) * slice_interval + slice_interval / 2 - 1)

        img_3d = np.load(self.img_name_list[img_index] + '_data.npy')
        # 切片方式暂定为切第一个维度
        img = img_3d[slice_index, :, :]

        # mask 即为img
        # img = Image.fromarray(img, mode="L")
        img = to_pil_image(to_tensor(img))
        mask = img
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        mask = mask.squeeze()

        return img, mask

    def __len__(self):
        # 切片数为16
        return len(self.img_name_list * 16)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
