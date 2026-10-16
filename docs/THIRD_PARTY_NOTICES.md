# Third-party notices

[🏠 MIMA](../README.md) · [📚 Documentation](README.md)

> [!IMPORTANT]
> Original MIMA code is distributed under [GNU GPL v3.0](../LICENSE.md).
> The components below retain their original notices and license terms. The
> repository-level license does not replace component-specific terms.

| Component | Bundled location | Upstream | Retained license |
| --- | --- | --- | --- |
| InternVL-derived model, training and inference code | `MLLM/internvl_chat/`, `MLLM/teacher_student/vendor/internvl_chat/`, related `MLLM/segmentation/` code | [OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL) | [MIT text](../LICENSES/InternVL-MIT.txt); file-level notices govern incorporated components |
| LLaVA-derived code | `MLLM/internvl_chat_llava/` | [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) | [Apache 2.0](../MLLM/internvl_chat_llava/LICENSE) |
| LMDeploy | `MLLM/lmdeploy/` | [InternLM/lmdeploy](https://github.com/InternLM/lmdeploy) | [Apache 2.0](../MLLM/lmdeploy/LICENSE) |
| Point-LIO | `src/point_lio_crimson/` | [hku-mars/Point-LIO](https://github.com/hku-mars/Point-LIO) | [GPL v2](../src/point_lio_crimson/LICENSE) |
| IKFoM | `src/point_lio_crimson/include/IKFoM/` | [hku-mars/IKFoM](https://github.com/hku-mars/IKFoM) | [GPL v2](../src/point_lio_crimson/include/IKFoM/LICENSE) |
| ikd-Tree | `src/point_lio_crimson/include/ikd-Tree/` | [hku-mars/ikd-Tree](https://github.com/hku-mars/ikd-Tree) | [GPL v2](../src/point_lio_crimson/include/ikd-Tree/LICENSE) |
| DynamixelSDK 3.7.51 | `src/motor_interface/dynamixel/dynamixel_sdk/` | [ROBOTIS-GIT/DynamixelSDK](https://github.com/ROBOTIS-GIT/DynamixelSDK/tree/3.7.51) | [Apache 2.0](../src/motor_interface/dynamixel/dynamixel_sdk/LICENSE) |
| libb64 | `src/third_party/libb64/` | [libb64/libb64](https://github.com/libb64/libb64) | [Public-domain dedication](../src/third_party/libb64/LICENSE.md) |
| WebSocket++ | `src/third_party/websocketpp/` | [zaphoyd/websocketpp](https://github.com/zaphoyd/websocketpp) | [BSD-style license and bundled notices](../src/third_party/websocketpp/COPYING) |

InternVL source headers identify OpenGVLab's MIT license. LLaVA, LMDeploy and
model components with their own notices retain those terms. The local
`internvl_chat` package classifier has been aligned with its MIT source headers.
Point-LIO's package license field has been aligned with its bundled GPL v2 text.
These metadata corrections do not replace or remove upstream copyright notices.

<a id="models-and-datasets"></a>

## 📦 Models and datasets

Pretrained backbones, model weights and external datasets retain the terms
published by their respective providers. Code licenses do not automatically
license a model's training data or externally hosted checkpoints. MIMA model
identities and release availability are recorded in [MODEL_CARD.md](MODEL_CARD.md)
and the [requirement-vector model card](../MLLM/mima_requirement_vector/MODEL_CARD.md).

<a id="external-runtime-dependencies"></a>

## 🧩 External runtime dependencies

ROS, PyTorch, MuJoCo, Gymnasium, NumPy, SciPy, OpenCV, PCL, Eigen, ZBar, libcurl,
yaml-cpp and nlohmann/json are installed dependencies rather than relicensed
MIMA source. Livox drivers are installed separately from their vendor repositories.
Their own terms apply.
