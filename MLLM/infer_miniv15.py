import argparse
from pathlib import Path

from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from PIL import Image

from torchvision.transforms.functional import InterpolationMode
import sys
import numpy as np
from pyntcloud import PyntCloud

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
npoints = 2048
sample_points_num = npoints


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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


def dynamic_preprocess(
    image, min_num=1, max_num=6, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def pc_norm(pc):
    pc = np.asarray(pc, dtype=np.float32)
    centered = pc - np.mean(pc, axis=0)
    scale = np.max(np.linalg.norm(centered, axis=1))
    return centered if scale < 1e-6 else centered / scale


def random_sample(pc, num):
    if len(pc) == 0:
        raise ValueError("point cloud is empty")
    indices = np.random.choice(len(pc), size=num, replace=len(pc) < num)
    return pc[indices]


def load_point_cloud(file_path):
    points = np.asarray(PyntCloud.from_file(file_path).xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("point cloud must contain xyz columns")
    points = points[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    points = pc_norm(random_sample(points, sample_points_num))
    return torch.from_numpy(points).float()


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "internvl_chat"))


def main():
    parser = argparse.ArgumentParser(
        description="Run the Mini-InternVL RGB/point-cloud example."
    )
    parser.add_argument(
        "--model", required=True, help="Checkpoint directory or model ID."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--point-cloud", type=Path, required=True)
    parser.add_argument("--instruction", default="Describe the image in detail.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("This example requires a CUDA device")
    for path in (args.image, args.point_cloud):
        if not path.expanduser().is_file():
            parser.error("Input file not found: %s" % path)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_path = (
        str(Path(args.model).expanduser()) if args.model.startswith("~") else args.model
    )
    model = (
        AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    pixels = load_image(args.image.expanduser(), max_num=6).to(dtype).cuda()
    points = load_point_cloud(str(args.point_cloud.expanduser()))
    points = points.expand(pixels.size(0), -1, -1).unsqueeze(0).to(dtype).cuda()
    response = model.chat(
        tokenizer,
        pixels,
        args.instruction,
        dict(num_beams=1, max_new_tokens=512, do_sample=False),
        points,
    )
    print(response)


if __name__ == "__main__":
    main()
