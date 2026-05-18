import random

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform


class HorizontalCrop(ImageOnlyTransform):
    def __init__(self, fov=90, always_apply=True, variance=10):

        super(HorizontalCrop, self).__init__(always_apply)
        self.mean_width_ratio = fov / 360
        self.variance = variance

    def apply(self, image, **params):
        random_width_ratio = self.mean_width_ratio + self.varience * random.uniform(
            -1, 1
        )
        random_crop_start = random.uniform(0, 1 - random_width_ratio) * image.shape[1]
        if self.size:
            image = image[
                :,
                int(random_crop_start) : int(
                    random_crop_start + random_width_ratio * image.shape[1]
                ),
                :,
            ]

        return image


class Cut(ImageOnlyTransform):
    def __init__(self, cutting=None, always_apply=False, p=1.0):

        super(Cut, self).__init__(always_apply, p)
        self.cutting = cutting

    def apply(self, image, **params):

        if self.cutting:
            image = image[self.cutting : -self.cutting, :, :]

        return image

    def get_transform_init_args_names(self):
        return ("size", "cutting")


def get_transforms_train(
    image_size_sat,
    img_size_ground,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    ground_cutting=0,
    FoV_90=False,
):

    satellite_transforms = A.Compose(
        [
            A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
            A.Resize(
                image_size_sat[0],
                image_size_sat[1],
                interpolation=cv2.INTER_LINEAR_EXACT,
                p=1.0,
            ),
            A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.15,
                always_apply=False,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.AdvancedBlur(p=1.0),
                    A.Sharpen(p=1.0),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.GridDropout(ratio=0.4, p=1.0),
                    A.CoarseDropout(
                        max_holes=25,
                        max_height=int(0.2 * image_size_sat[0]),
                        max_width=int(0.2 * image_size_sat[0]),
                        min_holes=10,
                        min_height=int(0.1 * image_size_sat[0]),
                        min_width=int(0.1 * image_size_sat[0]),
                        p=1.0,
                    ),
                ],
                p=0.3,
            ),
            A.Normalize(mean, std),
            ToTensorV2(),
        ]
    )

    ground_transforms = A.Compose(
        [
            Cut(cutting=ground_cutting, p=1.0),
            ## adding FoV augmentation for testing against NIWA data
            (
                HorizontalCrop(fov=90, always_apply=True, variance=10)
                if FoV_90
                else A.NoOp()
            ),
            A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
            A.Resize(
                img_size_ground[0],
                img_size_ground[1],
                interpolation=cv2.INTER_LINEAR_EXACT,
                p=1.0,
            ),
            A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.15,
                always_apply=False,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.AdvancedBlur(p=1.0),
                    A.Sharpen(p=1.0),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.GridDropout(ratio=0.5, p=1.0),
                    A.CoarseDropout(
                        max_holes=25,
                        max_height=int(0.2 * img_size_ground[0]),
                        max_width=int(0.2 * img_size_ground[0]),
                        min_holes=10,
                        min_height=int(0.1 * img_size_ground[0]),
                        min_width=int(0.1 * img_size_ground[0]),
                        p=1.0,
                    ),
                ],
                p=0.3,
            ),
            A.Normalize(mean, std),
            ToTensorV2(),
            # if FoV_90
            # randomly crop (mean 90 degree varience += 10 degree) a (90+-10)/360 of the image
            A.RandomResizedCrop(
                height=img_size_ground[0],
                width=(
                    img_size_ground[1] if not FoV_90 else int(img_size_ground[1] * 0.25)
                ),
                scale=(0.22, 0.28) if FoV_90 else (1.0, 1.0),
                ratio=(1.0, 1.0),
                interpolation=cv2.INTER_LINEAR_EXACT,
                p=1.0,
            ),
        ]
    )

    return satellite_transforms, ground_transforms


def get_transforms_val(
    image_size_sat,
    img_size_ground,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    ground_cutting=0,
    FoV_90=False,
):

    satellite_transforms = A.Compose(
        [
            A.Resize(
                image_size_sat[0],
                image_size_sat[1],
                interpolation=cv2.INTER_LINEAR_EXACT,
                p=1.0,
            ),
            A.Normalize(mean, std),
            ToTensorV2(),
        ]
    )

    ground_transforms = A.Compose(
        [
            Cut(cutting=ground_cutting, p=1.0),
            ## adding FoV augmentation for testing against NIWA data
            (
                HorizontalCrop(fov=90, always_apply=True, variance=10)
                if FoV_90
                else A.NoOp()
            ),
            A.Resize(
                img_size_ground[0],
                img_size_ground[1],
                interpolation=cv2.INTER_LINEAR_EXACT,
                p=1.0,
            ),
            A.Normalize(mean, std),
            ToTensorV2(),
        ]
    )

    return satellite_transforms, ground_transforms
