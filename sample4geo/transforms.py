import random

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform


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
):

    satellite_transforms = A.Compose(
        [
            A.ImageCompression(quality_range=(80,100), p=0.5),
            A.Resize(
                image_size_sat[0],
                image_size_sat[1],
                interpolation=cv2.INTER_LINEAR_EXACT,
                p=1.0,
            ),
            A.ColorJitter(
                brightness_range=(0.85, 1.15),
                contrast_range=(0.85, 1.15),
                saturation_range=(0.85, 1.15),
                hue_range=(-0.15, 0.15),
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
                        num_holes_range=(10,25),
                        hole_height_range=(int(0.1 * image_size_sat[0]), int(0.2 * image_size_sat[0])),
                        hole_width_range=(int(0.1 * image_size_sat[0]), int(0.2 * image_size_sat[0])),
                        p=0.85,
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
            A.ImageCompression(quality_range=(80,100), p=0.5),
            A.Resize(
                img_size_ground[0],
                img_size_ground[1],
                interpolation=cv2.INTER_LINEAR_EXACT,
                p=1.0,
            ),
            A.ColorJitter(
                brightness_range=(0.85, 1.15),
                contrast_range=(0.85, 1.15),
                saturation_range=(0.85, 1.15),
                hue_range=(-0.15, 0.15),
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
                        num_holes_range=(10,25),
                        hole_height_range=(int(0.1 * image_size_sat[0]), int(0.2 * image_size_sat[0])),
                        hole_width_range=(int(0.1 * image_size_sat[0]), int(0.2 * image_size_sat[0])),
                        p=0.85,
                    ),
                ],
                p=0.3,
            ),
            A.Normalize(mean, std),
            ToTensorV2()
        ]
    )

    return satellite_transforms, ground_transforms


def get_transforms_val(
    image_size_sat,
    img_size_ground,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    ground_cutting=0,
    # FoV_90=False,
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
