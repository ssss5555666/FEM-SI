import os

import torch
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
import glob
from torchvision.utils import save_image

def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):#判断是否为路径
        frames = sorted(os.listdir(name))#对路径下的文件和文件夹进行排序
        num_frames = len(frames)#文件个数
        video_array = np.array(#将数据转化为矩阵
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])#读取路径下的video文件
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):#如果后缀名为png或者jpg
        #lower：把字符串中的所有大写字母转换为小写字母
        #.endswith('.*')判断是否为指定后缀名
        image = io.imread(name)#读取文件

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)#灰度转RGB

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)#图片转为单精度浮点模式 256（0） 256（1） 3

        video_array = np.moveaxis(image, 1, 0)#改变形状256（1） 256（0） 3

        video_array = video_array.reshape((-1,) + frame_shape)#把形状改变为 *，256.1，256.2，3
        video_array = np.moveaxis(video_array, 1, 2)#* 256.2 256.1 3  对应前面的moveaxis
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        #如果是gif或mp4或mov
        video = np.array(mimread(name))#读取文件中多个图像，返回一个np数组
        if len(video.shape) == 3:#如果等于3
            video = np.array([gray2rgb(frame) for frame in video])#灰度转RGB
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
      视频数据集，每个视频可以表示为：
        -连接帧的图像
        -“.mp4”或“.gif”
        -包含所有帧的文件夹
    """

    def __init__(self, root_dir=r'H:\Song\FaceCycle-master\Imgs\original',root_dir_aligned=r'H:\Song\FaceCycle-master\Imgs\aligned22', frame_shape=(64, 64, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir#原图路径
        self.root_dir_aligned = root_dir_aligned#对齐脸路径
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)#使用tuple后和list不同，不能改变元素
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            # assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:#参数
                train_videos = {os.path.basename(video).split('#')[0] for video in#以#为分隔符，返回第一部分
                                os.listdir(os.path.join(root_dir, 'train'))}#返回数据集列表
                train_videos = list(train_videos)#转化为列表，列表是可以改动的
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))#
            # test_videos = os.listdir(os.path.join(root_dir, 'test'))#测试集
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')#训练路径
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform()#图像增强处理
        else:
            self.transform = None
        #print(self.videos)
    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:#如果都为Ture，即数据集为mp4
            name = self.videos[idx]#video名称
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))#随机选取video（*.mp4）
        else:#数据集为图像帧
            name = self.videos[idx]#
            path = os.path.join(self.root_dir, name)#每个图像帧文件夹的路径（原图
            #print(path,'555')
            path_aligned = os.path.join(self.root_dir_aligned, 'train')#每个对齐脸文件夹路径
            path_aligned = os.path.join(path_aligned,name)
            #print(path_aligned)

        video_name = os.path.basename(path)#返回每个视频数据集的名称
        #print(video_name)
        # 排序，最小的就是source，最大的就是driving_C,然后用他的随机函数取一张，driving_B
        if self.is_train and os.path.isdir(path_aligned):#如果为T并且path存在
            frames = os.listdir(path_aligned)#返回数据集下每张图片
            num_frames = len(frames)#图片张数
            frame_idx = [0,1,2]#
            if num_frames > 8:
                frame_idx[0] = 0
                frame_idx[1] = np.random.randint(3, num_frames // 2 + 1)
                frame_idx[2] = np.random.randint(num_frames - 3, num_frames)
                #
                # frame_idx[1] = np.random.randint(3, num_frames - 1)
                # frame_idx[2] = np.random.randint(num_frames - 1, num_frames)
            else:
                frame_idx[0] = 0
                frame_idx[1] = num_frames // 2
                frame_idx[2] = num_frames - 1
            #video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]#读取原图图片
            video_array_aligned = [img_as_float32(io.imread(os.path.join(path_aligned, frames[idx]))) for idx in frame_idx]#读取对齐脸图片
            #video_array_total = np.concatenate((video_array, video_array_aligned))

            #print(os.path.join(path, frames[0])\
            #      ,os.path.join(path, frames[1])\
            #      ,os.path.join(path, frames[2]))
            #print(os.path.join(path_aligned, frames[0]) \
            #       , os.path.join(path_aligned, frames[1]) \
            #       , os.path.join(path_aligned, frames[2]))
            #video_array1 = [os.path.join(path, frames[idx]) for idx in frame_idx]
            # print(path, '555')
            # print(video_array[0].shape,video_array[1].shape,video_array[2].shape)
            # print(path_aligned)
            # print(video_array_aligned[0].shape, video_array_aligned[1].shape, video_array_aligned[2].shape)

        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                num_frames)
            video_array = video_array[frame_idx]

        if self.transform is not None:
            video_array_aligned = self.transform(video_array_aligned)#做增强处理
            #video_array_aligned[0] = self.transform(video_array_aligned[0])
            #video_array_aligned[1] = self.transform(video_array_aligned[1])
            #video_array_aligned[2] = self.transform(video_array_aligned[2])


        out = {}
        if self.is_train:
            #onset_original = np.array(video_array_total[0], dtype='float32')#原图初始帧
            #mid_original = np.array(video_array_total[1], dtype='float32')#原图中间帧
            #apex_original = np.array(video_array_total[2], dtype='float32')#原图顶帧
            onset_aligned = np.array(video_array_aligned[0], dtype='float32')  # 对齐脸初始帧
            mid_aligned = np.array(video_array_aligned[1], dtype='float32')  #对齐脸中间帧
            apex_aligned = np.array(video_array_aligned[2], dtype='float32')#对齐脸顶帧

            #print(onset_original.shape,'shape')
            #out['onset_original'] = onset_original.transpose((2, 0, 1))
            #out['mid_original'] = mid_original.transpose((2, 0, 1))
            #out['apex_original'] = apex_original.transpose((2, 0, 1))
            out['onset_aligned'] = onset_aligned.transpose((2, 0, 1))
            out['mid_aligned'] = mid_aligned.transpose((2, 0, 1))
            out['apex_aligned'] = apex_aligned.transpose((2, 0, 1))

        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        return out



class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    在同一数据集上多次传递以获得更好的 I/O 性能
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    用于动画的成对数据集
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)#设置seed

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T# 2 nx*ny  转置后为 nx*ny 2
            number_of_pairs = min(xy.shape[0], number_of_pairs)#取小
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)#随机选择
        else:
            videos = self.initial_dataset.videos#video
            name_to_index = {name: index for index, name in enumerate(videos)}#
            pairs = pd.read_csv(pairs_list)#读取数据并且转换成dataframe数据帧
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]#返回布尔索引

            number_of_pairs = min(pairs.shape[0], number_of_pairs)#取小
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}

        return {**first, **second}




