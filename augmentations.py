import cv2
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomGamma, ElasticTransform, ChannelShuffle,RGBShift, Rotate,
    Normalize
)

from albumentations.torch import ToTensor


def strong_aug(p=1):
    return Compose([
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=.1),
            Blur(blur_limit=3, p=.1),
        ], p=0.2),
        #ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.3),
        #HueSaturationValue(p=0.3),
    ], p=p)




def get_transforms(phase, cfg):
    size = cfg['size']
    mean = eval(cfg['mean'])
    std = eval(cfg['std'])

    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                Transpose(p=0.5),
                Flip(p=0.5),
                ShiftScaleRotate(
                    shift_limit=0,  # no resizing
                    scale_limit=0.1,
                    rotate_limit=120,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),
                #RandomBrightnessContrast(p=0.25),
                strong_aug(),
            ]
        )
    list_transforms.extend(
        [

            Normalize(mean=mean, std=std, p=1),
            #Resize(size, size),
            ToTensor(normalize=None),  # [6]
        ]
    )
    return Compose(list_transforms)


def get_test_transforms(size, mean, std):

    list_transforms = []
    list_transforms.extend(
        [
            strong_aug(),
            #ShiftScaleRotate(
            #    shift_limit=0,  # no resizing
            #    scale_limit=0.1,
            #    rotate_limit=120,
            #    p=0.5,
            #    border_mode=cv2.BORDER_CONSTANT
            #),
        ]
    )
    return Compose(list_transforms)


'''
        self.TTA = albumentations.Compose(
            [
                albumentations.Rotate(limit=180, p=0.5),
                albumentations.Transpose(p=0.5),
                albumentations.Flip(p=0.5),
                albumentations.RandomScale(scale_limit=0.1),
            ]
        )
        self.transform = albumentations.Compose(
            [
                albumentations.Normalize(mean=mean, std=std, p=1),
                albumentations.Resize(size, size),
                AT.ToTensor(),
            ]
        )


'''
