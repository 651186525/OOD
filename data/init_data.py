import glob
import os
import random

import numpy as np
from tqdm import tqdm


# 随机将整个数据集划分，将信息写入txt文件中
def split_data(files_path):
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)

    random.seed(0)  # 设置随机种子，保证随机结果可复现
    val_rate = 0.2
    test_ = False

    # 获取所有json名称
    files_name = [file for file in os.listdir(files_path) if file.endswith('_data.npy')]
    random.shuffle(files_name)
    files_num = len(files_name)
    val_index = random.sample(range(0, files_num), k=int(files_num * val_rate))
    train_files = []
    val_files = []
    if test_ is True:
        test_index = random.sample(val_index, k=int(len(val_index) * 0.5))
        test_files = []
    for index, file_name in enumerate(files_name):
        if test_ is True and index in test_index:
            test_files.append(file_name.split('_data.npy')[0])
        elif index in val_index:
            val_files.append(file_name.split('_data.npy')[0])
        else:
            train_files.append(file_name.split('_data.npy')[0])

    try:
        train_f = open("brain/train.txt", "x")
        eval_f = open("brain/val.txt", "x")
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
        if test_ is True:
            test_f = open('brain/test.txt', 'x')
            test_f.write('\n'.join(test_files))
    except FileExistsError as e:
        print(e)
        exit(1)


def nifti_to_numpy(input_folder: str, output_folder: str):
    """Converts all nifti files in a input folder to numpy and saves the data and affine matrix into the output folder

    Args:
        input_folder (str): Folder to read the nifti files from
        output_folder (str): Folder to write the numpy arrays to
    """
    import nibabel as nib
    for fname in tqdm(sorted(os.listdir(input_folder))):

        if not fname.endswith("nii.gz"):
            continue

        n_file = os.path.join(input_folder, fname)
        nifti = nib.load(n_file)

        np_data = nifti.get_fdata()
        np_affine = nifti.affine

        f_basename = fname.split(".")[0]

        np.save(os.path.join(output_folder, f_basename + "_data.npy"), np_data.astype(np.float16))
        np.save(os.path.join(output_folder, f_basename + "_aff.npy"), np_affine)


if __name__ == '__main__':
    root = os.getcwd()
    input_dir = os.path.join(root, 'brain', 'brain_train')
    # 划分数据集
    split_data(input_dir)

    # shift_input_dir = '../../../my_mood/data/brain_toy/toy_label/'
    # shift_output_dir = '../../../my_mood/data/brain/toy_label/'
    # nifti_to_numpy(shift_input_dir, shift_output_dir)
