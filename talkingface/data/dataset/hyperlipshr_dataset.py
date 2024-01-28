import torch
import numpy as np
from talkingface.data.dataset.dataset import Dataset
from os.path import dirname, join, basename, isfile
from glob import glob
from talkingface.data.dataprocess.hyperlipsbase_process import HyperLipsBaseAudio
import os, random, cv2


class HyperLipsHRDataset(Dataset):
    def __init__(self, config, datasplit):
        super().__init__(config, datasplit)
        # 用的时候一定要把preprocessed_root的路径删去imgs
        gt_img_root = os.path.join(self.config['preprocessed_root'], 'HR_Train_Dateset', 'GT_IMG')
        self.gt_img = self.get_image_list(gt_img_root, datasplit)

        # 这个方法接受一个 frame 参数，通常是图像文件的路径，然后解析出帧的标识（通常是文件名中的数字部分）并返回一个整数

    def get_image_list(self, data_root, split_path):
        filelist = []
        print("split path:")
        print(split_path)
        with open(split_path) as f:
            for line in f:
                line = line.strip()
                if ' ' in line: line = line.split()[0]
                filelist.append(os.path.join(data_root, line))

        return filelist

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    # 这个方法接受一个 start_frame 参数，通常是一个起始帧的路径。它用于获取一个时间窗口内的图像帧序列，时间窗口的大小由全局变量 syncnet_T 决定
    # 通过分析 start_frame 的帧标识来确定窗口内的图像帧，并构建一个包含这些帧文件路径的列表 window_fnames。
    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + self.config['syncnet_T']):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    # 接受一个 window_fnames 参数，它是一个图像帧文件路径的列表。这些方法用于读取这些图像帧文件，并进行一些预处理操作。
    # read_window 方法将图像帧的大小调整为 args.img_size，然后返回图像列表。
    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (self.config['img_size'], self.config['img_size']))
            except Exception as e:
                return None

            window.append(img)

        return window

    # read_window_base 方法将图像帧的大小调整为固定的 128x128，然后返回图像列表
    def read_window_base(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (128, 128))
            except Exception as e:
                return None

            window.append(img)

        return window

    # 接受一个window_fnames参数，用于读取图像帧文件并进行预处理，通常用于生成"sketch"（草图）图像
    # 会根据全局变量args.img_size的值来确定不同的卷积核大小和图像大小，并对图像进行模糊处理和二值化处理
    def read_window_sketch(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                if self.config['img_size'] == 128:
                    kenerl_size = 5
                elif self.config['img_size'] == 256:
                    kenerl_size = 7
                elif self.config['img_size'] == 512:
                    kenerl_size = 11
                else:
                    print("Please input rigtht img_size!")
                img = cv2.resize(img, (self.config['img_size'], self.config['img_size']))
                img = cv2.GaussianBlur(img, (kenerl_size, kenerl_size), 0, 0, cv2.BORDER_DEFAULT)
                ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
            except Exception as e:
                return None

            window.append(img)

        return window

    def read_window_sketch_base(self, window_fnames):
        if window_fnames is None: return None
        window = []
        img_size = 128
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                if img_size == 128:
                    kenerl_size = 5
                elif img_size == 256:
                    kenerl_size = 7
                elif img_size == 512:
                    kenerl_size = 11
                else:
                    print("Please input rigtht img_size!")
                img = cv2.resize(img, (img_size, img_size))
                img = cv2.GaussianBlur(img, (kenerl_size, kenerl_size), 0, 0, cv2.BORDER_DEFAULT)
                ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
            except Exception as e:
                return None

            window.append(img)

        return window

    # 接受一个 window_fnames 参数，用于读取图像帧文件并提取坐标信息。
    # 它会查找图像中像素值为255的坐标，并返回坐标的最大和最小值
    def read_coord(self, window_fnames):
        if window_fnames is None: return None

        coords = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (self.config['img_size'], self.config['img_size']))
            except Exception as e:
                return None
            index = np.argwhere(img[:, :, 0] == 255)
            x_max = max(index[:, 0])
            x_min = min(index[:, 0])
            y_max = max(index[:, 1])
            y_min = min(index[:, 1])
            coords.append([x_min, x_max, y_min, y_max])
        return coords

    # 接受一个 window 参数，通常是一个图像帧列表。它对图像进行一些准备工作，包括将图像值缩放到0到1之间，并将通道维度移动到第一维
    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.gt_img)  # 返回数据集中样本的数量，通常是 gt_img 列表的长度

    # 获取数据集中的样本。
    # 它首先选择一个随机索引 idx，然后根据该索引构建图像和标签的路径。
    # 接着，它调用上述方法来获取图像数据、sketch数据、mask数据等，并将它们返回
    def __getitem__(self, idx):
        while 1:

            idx = random.randint(0, len(self.gt_img) - 1)
            # vidname = os.path.join(self.gt_img[idx].split('/')[-2],self.gt_img[idx].split('/')[-1])       # 在此处有修改
            vidname = os.path.join(self.gt_img[idx].split('\\')[-2], self.gt_img[idx].split('\\')[-1])
            gt_img_root = os.path.join(self.config['preprocessed_root'], 'HR_Train_Dateset', 'GT_IMG')
            gt_sketch_data_root = os.path.join(self.config['preprocessed_root'], 'HR_Train_Dateset', 'GT_SKETCH')
            gt_mask_root = os.path.join(self.config['preprocessed_root'], 'HR_Train_Dateset', 'GT_MASK')
            hyper_img_root = os.path.join(self.config['preprocessed_root'], 'HR_Train_Dateset', 'HYPER_IMG')
            hyper_sketch_data_root = os.path.join(self.config['preprocessed_root'], 'HR_Train_Dateset', 'HYPER_SKETCH')

            gt_img_names = list(glob(join(gt_img_root, vidname, '*.jpg')))
            gt_sketch_names = list(glob(join(gt_sketch_data_root, vidname, '*.jpg')))
            gt_mask_names = list(glob(join(gt_mask_root, vidname, '*.jpg')))
            hyper_img_names = list(glob(join(hyper_img_root, vidname, '*.jpg')))
            hyper_sketch_names = list(glob(join(hyper_sketch_data_root, vidname, '*.jpg')))
            if not (len(gt_img_names) == len(gt_sketch_names) == len(gt_mask_names) == len(hyper_img_names) == len(
                    hyper_sketch_names)):
                continue
            if len(gt_img_names) <= 3 * self.config['syncnet_T']:
                continue

            # img_name = random.choice(gt_img_names).split('/')[-1]     # 这一行有修改
            img_name = random.choice(gt_img_names).split('\\')[-1]
            gt_img_name = join(gt_img_root, vidname, img_name)
            gt_sketch_name = join(gt_sketch_data_root, vidname, img_name)
            gt_mask_name = join(gt_mask_root, vidname, img_name)
            hyper_img_name = join(hyper_img_root, vidname, img_name)
            hyper_sketch_name = join(hyper_sketch_data_root, vidname, img_name)

            gt_img_name_window_frames = self.get_window(gt_img_name)
            gt_sketch_name_window_frames = self.get_window(gt_sketch_name)
            gt_mask_name_window_frames = self.get_window(gt_mask_name)
            hyper_img_name_window_frames = self.get_window(hyper_img_name)
            hyper_sketch_name_window_frames = self.get_window(hyper_sketch_name)

            coords = self.read_coord(gt_mask_name_window_frames)

            if gt_img_name_window_frames is None:
                continue

            gt_img_window = self.read_window(gt_img_name_window_frames)
            gt_sketch_window = self.read_window_sketch(gt_sketch_name_window_frames)
            gt_mask_window = self.read_window(gt_mask_name_window_frames)
            hyper_img_window = self.read_window_base(hyper_img_name_window_frames)
            hyper_sketch_window = self.read_window_sketch_base(hyper_sketch_name_window_frames)

            gt_img_window = self.prepare_window(gt_img_window)
            gt_sketch_window = self.prepare_window(gt_sketch_window)
            gt_mask_window = self.prepare_window(gt_mask_window)
            hyper_img_window = self.prepare_window(hyper_img_window)
            hyper_sketch_window = self.prepare_window(hyper_sketch_window)

            gt_img = torch.FloatTensor(gt_img_window)
            gt_sketch = torch.FloatTensor(gt_sketch_window)
            gt_mask = torch.FloatTensor(gt_mask_window)
            hyper_img = torch.FloatTensor(hyper_img_window)
            hyper_sketch = torch.FloatTensor(hyper_sketch_window)
            coords = torch.FloatTensor(coords)
            # return gt_img, gt_sketch, gt_mask, hyper_img, hyper_sketch, coords

            return {'gt_img': gt_img, 'gt_sketch': gt_sketch, 'gt_mask': gt_mask, 'hyper_img': hyper_img,
                    'hyper_sketch': hyper_sketch, 'coords': coords}
