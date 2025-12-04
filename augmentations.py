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
    """
    Data augmentations for VICReg.

    Returns: (view1, view2)
    Both are 96x96 crops with strong color + blur + (optional) solarization.
    """
    def __init__(self, size=96, s=1.0):
        # ---- Color jitter + grayscale (what you already had) ----
        color_jitter = transforms.ColorJitter(
            0.8 * s,  # brightness
            0.8 * s,  # contrast
            0.8 * s,  # saturation
            0.2 * s,  # hue
        )
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)

        # ---- Common spatial transforms ----
        # For 96x96 images, a scale of (0.2, 1.0) is usually safe.
        # If crops look too small, you can change to (0.3, 1.0).
        self.base_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size,
                scale=(0.2, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            rnd_color_jitter,
            rnd_gray,
        ])

        # Normalization (ImageNet stats – standard for ResNet backbones)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # ---- View 1: strong blur, no solarization ----
        self.transform1 = transforms.Compose([
            self.base_transform,
            GaussianBlur(p=1.0),
            transforms.ToTensor(),
            normalize,
        ])

        # ---- View 2: weaker blur + some solarization ----
        self.transform2 = transforms.Compose([
            self.base_transform,
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    def __call__(self, x):
        v1 = self.transform1(x)
        v2 = self.transform2(x)
        return v1, v2
