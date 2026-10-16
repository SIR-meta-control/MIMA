# MLLM Installation

[🏠 MIMA](../README.md) · [📚 Documentation](../docs/README.md)

> [!NOTE]
> These dependencies apply to the InternVL code included in `MLLM/`.
> Run the installation commands in a Linux environment with CUDA.
> The package versions below describe the supplied implementation.

## 📋 Table of Contents

- [🐍 1. Create the Environment](#1-create-the-environment)
- [🔥 2. Install PyTorch and CUDA](#2-install-pytorch-and-cuda)
- [⚡ 3. Install FlashAttention](#3-install-flashattention)
- [👁️ 4. Install Vision Dependencies](#4-install-vision-dependencies)
- [🧠 5. Install Transformers](#5-install-transformers)
- [🧩 6. Install Apex (Optional)](#6-install-apex-optional)
- [🧩 7. Install Other Requirements](#7-install-other-requirements)

<a id="1-create-the-environment"></a>

## 🐍 1. Create the Environment

```bash
conda create -n internvl python=3.9 -y
conda activate internvl
```

<a id="2-install-pytorch-and-cuda"></a>

## 🔥 2. Install PyTorch and CUDA

This implementation requires `PyTorch>=2.0`, `torchvision>=0.15.2` and
`CUDA>=11.6`.

For examples, to install `torch==2.0.1` with `CUDA==11.8`:

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# or
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

<a id="3-install-flashattention"></a>

## ⚡ 3. Install FlashAttention

```bash
pip install flash-attn==2.3.6 --no-build-isolation
```

Alternatively you can compile from source:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.3.6
python setup.py install
```

<a id="4-install-vision-dependencies"></a>

## 👁️ 4. Install Vision Dependencies

```bash
pip install timm==0.9.12
pip install -U openmim
mim install mmcv-full==1.6.2  # (optional, for mmsegmentation)
```

<a id="5-install-transformers"></a>

## 🧠 5. Install Transformers

```bash
pip install transformers==4.37.2
```

<a id="6-install-apex-optional"></a>

## 🧩 6. Install Apex (Optional)

```bash
git clone https://github.com/NVIDIA/apex.git
git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82  # https://github.com/NVIDIA/apex/issues/1735
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

If you meet `ModuleNotFoundError: No module named 'fused_layer_norm_cuda'`, please note that this is because apex's CUDA extensions are not being installed successfully. You can try to uninstall apex and the code will default to the PyTorch version of RMSNorm; Or, if you want to use apex, try adding a few lines to `setup.py`, like this, and then recompiling.

<img src=https://github.com/OpenGVLab/InternVL/assets/23737120/c04a989c-8024-49fa-b62c-2da623e63729 width=50%>

<a id="7-install-other-requirements"></a>

## 🧩 7. Install Other Requirements

```bash
pip install "numpy<2" "opencv-python<4.12" "pyntcloud>=0.3,<0.4" termcolor yacs pyyaml scipy
pip install deepspeed==0.13.5
pip install pycocoevalcap tqdm
```
