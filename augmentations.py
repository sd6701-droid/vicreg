from PIL import ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image with probability p.
    Sigma is sampled in [0.1, 2.0].
    """
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    """
    Apply Solarization to the PIL image with probability p.
    """
    def __init__(self, p=0.0):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class TrainTransform(object):
    def __init__(self, size=96, s=1.0):
        color_jitter = transforms.ColorJitter(
            0.8 * s,
            0.8 * s,
            0.8 * s,
            0.2 * s,
        )
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)

        self.base_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size,
                scale=(0.2, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),

            # >>> added small random rotations <<<
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=10)],
                p=0.3,   # don’t rotate every image
            ),

            rnd_color_jitter,
            rnd_gray,
        ])

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.transform1 = transforms.Compose([
            self.base_transform,
            GaussianBlur(p=1.0),
            transforms.ToTensor(),
            normalize,
        ])

        self.transform2 = transforms.Compose([
            self.base_transform,
            GaussianBlur(p=0.1),
            Solarization(p=0.0),
            transforms.ToTensor(),
            normalize,
        ])

    def __call__(self, x):
        v1 = self.transform1(x)
        v2 = self.transform2(x)
        return v1, v2
