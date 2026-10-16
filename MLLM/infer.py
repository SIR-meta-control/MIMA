import argparse
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pyntcloud import PyntCloud
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NPOINTS = 2048


def build_transform(input_size):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


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
    if min_num < 1 or max_num < min_num:
        raise ValueError(
            f"invalid min/max blocks: min_num={min_num}, max_num={max_num}"
        )
    if orig_height == 0:
        raise ValueError("input image has zero height")
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    # Sort deterministically so behavior does not depend on set iteration order.
    target_ratios = sorted(target_ratios, key=lambda x: (x[0] * x[1], x[0], x[1]))

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    with Image.open(image_file) as image:
        image = image.convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def load_depth_image(depth_file, input_size=448, max_num=6):
    # Reuse image encoder preprocessing for depth frames converted to 3 channels.
    with Image.open(depth_file) as depth_img:
        depth_img = depth_img.convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        depth_img, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    depth_values = [transform(img) for img in images]
    return torch.stack(depth_values)


def pc_norm(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    if scale < 1e-6:
        return pc
    return pc / scale


def random_sample(pc, num):
    if pc.shape[0] == 0:
        raise ValueError("point cloud is empty")
    replace = pc.shape[0] < num
    indices = np.random.choice(pc.shape[0], size=num, replace=replace)
    return pc[indices]


def load_point_cloud(file_path):
    pcd = PyntCloud.from_file(file_path).xyz
    if hasattr(pcd, "to_numpy"):
        pcd = pcd.to_numpy()
    pcd = np.asarray(pcd, dtype=np.float32)
    if pcd.ndim != 2 or pcd.shape[0] == 0:
        raise ValueError(f"invalid point cloud shape from {file_path}: {pcd.shape}")
    if pcd.shape[1] < 3:
        raise ValueError(f"point cloud must have >=3 columns, got {pcd.shape[1]}")
    pcd = pcd[:, :3]
    finite_mask = np.isfinite(pcd).all(axis=1)
    pcd = pcd[finite_mask]
    if pcd.shape[0] == 0:
        raise ValueError(f"point cloud from {file_path} contains no finite xyz rows")
    pcd = random_sample(pcd, NPOINTS)
    pcd = pc_norm(pcd)
    return torch.from_numpy(pcd).float()


def parse_odometry(odom_text, expected_dim=None):
    values = [float(item.strip()) for item in odom_text.split(",") if item.strip()]
    if len(values) == 0:
        raise ValueError("odometry is empty, expected comma-separated floats")
    if expected_dim is not None and len(values) != expected_dim:
        raise ValueError(
            f"odometry dimension mismatch: expected {expected_dim}, got {len(values)}"
        )
    if not np.isfinite(values).all():
        raise ValueError("odometry contains non-finite values")
    return torch.tensor(values, dtype=torch.float32)


def ensure_special_tokens(tokenizer, model):
    needed = [
        "<PC_CONTEXT>",
        "<pointcloud>",
        "</pointcloud>",
        "<ODOM_CONTEXT>",
        "<odometry>",
        "</odometry>",
        "<DEPTH_CONTEXT>",
        "<depth>",
        "</depth>",
        "<MAP_CONTEXT>",
        "<map>",
        "</map>",
    ]
    missing = [
        tok
        for tok in needed
        if tokenizer.convert_tokens_to_ids(tok) == tokenizer.unk_token_id
    ]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        if hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))
        elif hasattr(model, "language_model") and hasattr(
            model.language_model, "resize_token_embeddings"
        ):
            model.language_model.resize_token_embeddings(len(tokenizer))
        else:
            raise AttributeError(
                "Model does not expose resize_token_embeddings for tokenizer extension"
            )


def resolve_cuda_dtype():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required because model.chat() in this codebase uses .cuda() internally"
        )
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def build_system_prompt(instruction):
    return f"""You are an embodied AI agent executing tasks in a real-world scene.
You will receive the following multimodal sensor inputs:
1. RGB image: current front-view camera observation.
2. Depth image: geometric depth cue aligned with the scene.
3. LiDAR point cloud: current local 3D observation.
4. LiDAR map point cloud: global or prior 3D map context.
5. Odometry: ego-motion/state vector.
6. Instruction: task requirement from user.

Your job:
- Analyze all sensor inputs jointly.
- Understand the scene context.
- Infer the most suitable task type required by the instruction.

Instruction: {instruction}

Output:
- scene description
- task type
"""


def main():
    parser = argparse.ArgumentParser(
        description="Infer with image + point cloud + odometry inputs"
    )
    parser.add_argument("--model-path", required=True, help="Local model path")
    parser.add_argument("--instruction", required=True, help="English instruction")
    parser.add_argument("--image-path", required=True, help="Input RGB image path")
    parser.add_argument("--depth-path", required=True, help="Input depth image path")
    parser.add_argument(
        "--point-cloud-path", required=True, help="Input lidar point cloud path (.ply)"
    )
    parser.add_argument(
        "--map-point-cloud-path",
        required=True,
        help="Input lidar map point cloud path (.ply)",
    )
    parser.add_argument(
        "--odometry",
        required=True,
        help="Comma-separated odometry vector, e.g. 0.1,0.2,0.3,0.0,0.0,1.57",
    )
    parser.add_argument(
        "--max-num",
        type=int,
        default=6,
        help="Maximum image tiles for dynamic preprocess",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="Generation max_new_tokens"
    )
    args = parser.parse_args()

    if args.max_num < 1:
        raise ValueError(f"--max-num must be >= 1, got {args.max_num}")

    compute_dtype = resolve_cuda_dtype()

    model = (
        AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    ensure_special_tokens(tokenizer, model)

    pixel_values = (
        load_image(args.image_path, max_num=args.max_num).to(compute_dtype).cuda()
    )

    depth_values = (
        load_depth_image(args.depth_path, max_num=args.max_num).to(compute_dtype).cuda()
    )

    if depth_values.size(0) != pixel_values.size(0):
        raise ValueError(
            f"Depth/image tile count mismatch: depth={depth_values.size(0)} image={pixel_values.size(0)}"
        )

    point_cloud = load_point_cloud(args.point_cloud_path)
    point_cloud = (
        point_cloud.expand(pixel_values.size(0), -1, -1)
        .unsqueeze(0)
        .to(compute_dtype)
        .cuda()
    )

    map_point_cloud = load_point_cloud(args.map_point_cloud_path)
    map_point_cloud = (
        map_point_cloud.expand(pixel_values.size(0), -1, -1)
        .unsqueeze(0)
        .to(compute_dtype)
        .cuda()
    )

    expected_odom_dim = getattr(model.config, "odometry_input_dim", None)
    odometry = parse_odometry(args.odometry, expected_dim=expected_odom_dim)
    odometry = (
        odometry.expand(pixel_values.size(0), -1).unsqueeze(0).to(compute_dtype).cuda()
    )

    generation_config = {
        "num_beams": 1,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
    }

    response = model.chat(
        tokenizer,
        pixel_values,
        build_system_prompt(args.instruction),
        generation_config,
        point_cloud=point_cloud,
        map_point_cloud=map_point_cloud,
        depth_values=depth_values,
        odometry=odometry,
    )
    print(args.instruction, response)


if __name__ == "__main__":
    main()
