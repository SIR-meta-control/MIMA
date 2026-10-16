from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from refactored_mllm.features import load_depth_array


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int, *, is_depth: bool = False) -> T.Compose:
    if is_depth:
        return T.Compose(
            [
                T.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
            ]
        )
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 6,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for index in range(blocks):
        box = (
            (index % (target_width // image_size)) * image_size,
            (index // (target_width // image_size)) * image_size,
            ((index % (target_width // image_size)) + 1) * image_size,
            ((index // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(
    image_file: str | Path, input_size: int = 448, max_num: int = 6
) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_num,
    )
    return torch.stack([transform(image) for image in images])


def load_depth(
    depth_file: str | Path, input_size: int = 448, max_num: int = 6
) -> torch.Tensor:
    path = Path(depth_file)
    if path.suffix.lower() == ".npy":
        image = _depth_array_to_image(load_depth_array(path))
    else:
        image = Image.open(path).convert("L")
    transform = build_transform(input_size=input_size, is_depth=True)
    depths = dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_num,
    )
    tensors = []
    for depth in depths:
        tensor = transform(depth)
        max_value = tensor.max()
        if max_value > 0:
            tensor = tensor / max_value
        if tensor.max() != tensor.min():
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        tensors.append(tensor)
    return torch.stack(tensors).unsqueeze(0)


def _depth_array_to_image(depth: np.ndarray) -> Image.Image:
    array = np.asarray(depth, dtype=float)
    finite = array[np.isfinite(array)]
    finite = finite[finite > 0]
    if finite.size == 0:
        normalized = np.zeros(array.shape, dtype=np.uint8)
    else:
        low = float(np.percentile(finite, 1))
        high = float(np.percentile(finite, 99))
        if high <= low:
            high = low + 1.0
        clipped = np.clip(array, low, high)
        clipped = np.nan_to_num(clipped, nan=high, posinf=high, neginf=low)
        normalized = ((clipped - low) / (high - low) * 255.0).astype(np.uint8)
    return Image.fromarray(normalized, mode="L")


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio
