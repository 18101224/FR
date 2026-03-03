from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


class KPRPEGeometricAugmentation:
    def __init__(
        self,
        scale_min: float = 0.8,
        scale_max: float = 1.2,
        rot_prob: float = 0.2,
        max_rot: float = 20.0,
        hflip_prob: float = 0.5,
        translate_ratio: float = 0.1,
    ) -> None:
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.rot_prob = float(rot_prob)
        self.max_rot = float(max_rot)
        self.hflip_prob = float(hflip_prob)
        self.translate_ratio = float(translate_ratio)

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.hflip_prob:
            image = F.hflip(image)

        width, height = image.size
        max_dx = int(round(width * self.translate_ratio))
        max_dy = int(round(height * self.translate_ratio))
        translate = (
            random.randint(-max_dx, max_dx) if max_dx > 0 else 0,
            random.randint(-max_dy, max_dy) if max_dy > 0 else 0,
        )
        scale = random.uniform(self.scale_min, self.scale_max)
        angle = random.uniform(-self.max_rot, self.max_rot) if random.random() < self.rot_prob else 0.0

        return F.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )


class KPRPEPhotometricRandAugment:
    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 14,
        magnitude_offset: int = 9,
        num_magnitude_bins: int = 31,
    ) -> None:
        self.num_ops = int(num_ops)
        self.magnitude = int(magnitude)
        self.magnitude_offset = int(magnitude_offset)
        self.num_magnitude_bins = int(num_magnitude_bins)
        self.op_meta = self._augmentation_space(self.num_magnitude_bins)
        self.op_names = list(self.op_meta.keys())

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Saturate": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Equalize": (torch.tensor(0.0), False),
            "Grayscale": (torch.tensor(0.0), False),
        }

    def _sample_op(self) -> Tuple[str, float]:
        op_name = random.choice(self.op_names)
        if op_name in {"Equalize", "Grayscale"}:
            op_name = random.choice(self.op_names)
            if op_name in {"Equalize", "Grayscale"}:
                op_name = random.choice(self.op_names)

        magnitudes, signed = self.op_meta[op_name]
        magnitude_idx = random.randint(
            self.magnitude - self.magnitude_offset,
            self.magnitude + self.magnitude_offset,
        )
        magnitude_idx = max(0, min(magnitude_idx, self.num_magnitude_bins - 1))
        if magnitudes.ndim > 0:
            magnitude = float(magnitudes[magnitude_idx].item())
        else:
            magnitude = 0.0
        if signed and random.random() < 0.5:
            magnitude *= -1.0
        return op_name, magnitude

    def _apply_op(self, image: Image.Image, op_name: str, magnitude: float) -> Image.Image:
        if op_name == "Brightness":
            return F.adjust_brightness(image, 1.0 + magnitude)
        if op_name == "Saturate":
            return F.adjust_saturation(image, 1.0 + magnitude)
        if op_name == "Contrast":
            return F.adjust_contrast(image, 1.0 + magnitude)
        if op_name == "Sharpness":
            return F.adjust_sharpness(image, 1.0 + magnitude)
        if op_name == "Equalize":
            return F.equalize(image)
        if op_name == "Grayscale":
            return F.to_grayscale(image, num_output_channels=3)
        return image

    def __call__(self, image: Image.Image) -> Image.Image:
        for _ in range(self.num_ops):
            op_name, magnitude = self._sample_op()
            image = self._apply_op(image, op_name, magnitude)
        return image


class KPRPEBlurAugmentation:
    def __init__(self, magnitude: float = 1.0, prob: float = 0.2) -> None:
        self.magnitude = float(magnitude)
        self.prob = float(prob)

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.prob:
            return image

        method = random.choice(
            ["box", "gaussian", "resize", "resize", "resize", "resize", "resize", "resize"]
        )
        if method == "box":
            radius = random.uniform(0.5, max(0.5, 3.0 * self.magnitude))
            return image.filter(ImageFilter.BoxBlur(radius=radius))
        if method == "gaussian":
            radius = random.uniform(0.1, max(0.1, 2.0 * self.magnitude))
            return image.filter(ImageFilter.GaussianBlur(radius=radius))

        width, height = image.size
        side_ratio = random.uniform(max(0.2, 1.0 - 0.8 * self.magnitude), 1.0)
        small_width = max(1, int(round(width * side_ratio)))
        small_height = max(1, int(round(height * side_ratio)))
        interpolation_down = random.choice(
            [
                InterpolationMode.NEAREST,
                InterpolationMode.BILINEAR,
                InterpolationMode.BICUBIC,
                InterpolationMode.LANCZOS,
            ]
        )
        interpolation_up = random.choice(
            [
                InterpolationMode.NEAREST,
                InterpolationMode.BILINEAR,
                InterpolationMode.BICUBIC,
                InterpolationMode.LANCZOS,
            ]
        )
        image = F.resize(image, [small_height, small_width], interpolation=interpolation_down)
        image = F.resize(image, [height, width], interpolation=interpolation_up)
        return image


class KPRPECutoutAugmentation:
    def __init__(self, prob: float = 0.2) -> None:
        self.prob = float(prob)

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.prob:
            return image

        image_np = np.array(image, copy=True)
        height, width = image_np.shape[:2]
        cutout_width = max(8, int(round(width * random.uniform(0.12, 0.28))))
        cutout_height = max(8, int(round(height * random.uniform(0.12, 0.28))))
        x0 = random.randint(0, max(0, width - cutout_width))
        y0 = random.randint(0, max(0, height - cutout_height))
        image_np[y0:y0 + cutout_height, x0:x0 + cutout_width] = 0
        return Image.fromarray(image_np)


class KPRPETrainAugmentation:
    """
    Matches the KP-RPE supplementary recipe and run_v1 augmentation order.
    """

    def __init__(self) -> None:
        self.geometric = KPRPEGeometricAugmentation()
        self.cutout = KPRPECutoutAugmentation(prob=0.2)
        self.blur = KPRPEBlurAugmentation(magnitude=1.0, prob=0.2)
        self.photometric = KPRPEPhotometricRandAugment(
            num_ops=2,
            magnitude=14,
            magnitude_offset=9,
            num_magnitude_bins=31,
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        image = self.geometric(image)
        image = self.cutout(image)
        image = self.blur(image)
        image = self.photometric(image)
        return image


def build_kprpe_train_transform():
    return transforms.Compose(
        [
            KPRPETrainAugmentation(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
