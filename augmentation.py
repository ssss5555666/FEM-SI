"""
Code from https://github.com/hassony2/torch_videovision
"""

import numbers

import random
import numpy as np
import PIL

from skimage.transform import resize, rotate
from skimage.util import pad
import torchvision

import warnings

from skimage import img_as_ubyte, img_as_float


def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):#判断
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]#剪裁

    elif isinstance(clip[0], PIL.Image.Image):#判断
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
            ]#对图片进行剪裁
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped#返回剪裁后的img


def pad_clip(clip, h, w):
    im_h, im_w = clip[0].shape[:2]#取前两个维度
    pad_h = (0, 0) if h < im_h else ((h - im_h) // 2, (h - im_h + 1) // 2)
    pad_w = (0, 0) if w < im_w else ((w - im_w) // 2, (w - im_w + 1) // 2)

    return pad(clip, ((0, 0), pad_h, pad_w, (0, 0)), mode='edge')#填充clip


def resize_clip(clip, size, interpolation='bilinear'):
    if isinstance(clip[0], np.ndarray):#如果clip[0]是数组
        if isinstance(size, numbers.Number):#如果size是numbers.Number
            im_h, im_w, im_c = clip[0].shape#提取clip[0]的形状
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):#如果符合
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)#更新h、w
            size = (new_w, new_h)#新size
        else:
            size = size[1], size[0]

        scaled = [
            resize(img, size, order=1 if interpolation == 'bilinear' else 0, preserve_range=True,
                   mode='constant', anti_aliasing=True) for img in clip#调整img大小
            ]
    elif isinstance(clip[0], PIL.Image.Image):#如果是图像
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.NEAREST
        else:
            pil_inter = PIL.Image.BILINEAR
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return scaled#返回调整过大小的img


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


class RandomFlip(object):
    def __init__(self, time_flip=False, horizontal_flip=True):
        self.time_flip = time_flip
        self.horizontal_flip = horizontal_flip

    def __call__(self, clip):
        if random.random() < 0.5 and self.time_flip:
            return clip[::-1]#原数组
        if random.random() < 0.5 and self.horizontal_flip:#random.random()用于生成一个0到1的随机符点数: 0 <= n < 1.0
            #clip[0] = np.fliplr(clip[0])
            return [np.fliplr(img) for img in clip]#在左右方向翻转图像数组

        return clip


class RandomResize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size
    The larger the original image is, the more times it takes to
    interpolate
    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
        将（H x W x C）numpy.ndarray列表的大小调整为最终大小
    原始图像越大，恢复所需的时间就越多
    插话
    Args：
    插值（str）：可以是“最近的”、“双线性”之一
    默认为最接近
    大小（元组）：（宽、高）
    """

    def __init__(self, ratio=(3. / 4., 4. / 3.), interpolation='nearest'):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, clip):
        scaling_factor = random.uniform(self.ratio[0], self.ratio[1])#随机选取(self.ratio[0], self.ratio[1])范围的数

        if isinstance(clip[0], np.ndarray):#如果是数组
            im_h, im_w, im_c = clip[0].shape#返回形状
        elif isinstance(clip[0], PIL.Image.Image):#如果是图片
            im_w, im_h = clip[0].size#返回形状

        new_w = int(im_w * scaling_factor)#返回w的int
        new_h = int(im_h * scaling_factor)#返回h的int
        new_size = (new_w, new_h)#新的形状
        resized = resize_clip(
            clip, new_size, interpolation=self.interpolation)#返回新size的clip

        return resized


class RandomCrop(object):
    """Extract random crop at the same location for a list of videos
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):#判断size
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of videos to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of videos
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):#判断clip【0】
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        clip = pad_clip(clip, h, w)#填充
        im_h, im_w = clip.shape[1:3]#新H、W
        x1 = 0 if h == im_h else random.randint(0, im_w - w)
        y1 = 0 if w == im_w else random.randint(0, im_h - h)
        cropped = crop_clip(clip, y1, x1, h, w)#剪裁clip

        return cropped#返回剪裁后的clip


class RandomRotation(object):
    """Rotate entire clip randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
        将整个剪辑随机旋转一个范围内的随机角度

    给定界限

    Args：

    度数（序列或整数）：可从中选择的度数范围

    如果度数是一个数字，而不是像（最小、最大）这样的序列，

    度的范围为（-degrees，+degrees）。
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):#判断degrees是否为numbers.Number的类型
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:#如果不是对应类型
            if len(degrees) != 2:#如果长度不等于2
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees#旋转角度

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of videos to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of videos
                Args：
        img（PIL.Image或numpy.ndarray）：要裁剪的视频列表
        格式（h、w、c）为numpy.ndarray
        返回：
        PIL.Image或numpy.ndarray：视频剪辑列表
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])#从(-degrees, degrees)范围随机生成一个角度
        if isinstance(clip[0], np.ndarray):#如果clip【0】是np.ndarray类型
            rotated = [rotate(image=img, angle=angle, preserve_range=True) for img in clip]
            #将clip中的img数组围绕其中心旋转angle角度
        elif isinstance(clip[0], PIL.Image.Image):#如果clip【0】是PIL.Image.Image类型
            rotated = [img.rotate(angle) for img in clip]#将图片旋转angle角度
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        return rotated


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.

    随机更改剪辑的亮度、对比度、饱和度和色调
    """

    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)#随机选取亮度系数
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)#随机选取对比系数
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)#随机选取饱和系数
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)#随机选取色相因子
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        if isinstance(clip[0], np.ndarray):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)
            img_transforms = [img_as_ubyte, torchvision.transforms.ToPILImage()] + img_transforms + [np.array,
                                                                                                     img_as_float]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                jittered_clip = []
                for img in clip:
                    jittered_img = img
                    for func in img_transforms:
                        jittered_img = func(jittered_img)
                    jittered_clip.append(jittered_img.astype('float32'))
        elif isinstance(clip[0], PIL.Image.Image):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all videos
            jittered_clip = []
            for img in clip:
                for func in img_transforms:
                    jittered_img = func(img)
                jittered_clip.append(jittered_img)

        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        return jittered_clip


class AllAugmentationTransform:
    def __init__(self, resize_param=None, rotation_param=None, flip_param=True, crop_param=None, jitter_param=True):
        self.transforms = []

        if flip_param is not None:
            self.transforms.append(RandomFlip(time_flip = False,horizontal_flip= True))#输入参数flip_param的函数，返回原数组或者左右翻转后的数组

        if rotation_param is not None:
            self.transforms.append(RandomRotation(**rotation_param))#返回旋转后的图片或者数组

        if resize_param is not None:
            self.transforms.append(RandomResize(**resize_param))#返回新的形状下的img

        if crop_param is not None:
            self.transforms.append(RandomCrop(**crop_param))#返回剪裁后的clip

        if jitter_param is not None:
            self.transforms.append(ColorJitter())#亮度，饱和，对比因子，色相因子

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip
