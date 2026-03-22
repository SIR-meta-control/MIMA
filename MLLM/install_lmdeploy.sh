export LMDEPLOY_VERSION=0.4.2
export PYTHON_VERSION=39
pip install https://github.com/InternLM/lmdeploy/releases/download/v${LMDEPLOY_VERSION}/lmdeploy-${LMDEPLOY_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux2014_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118

--model_name_or_path "/fuxi_team14/users/haoyanghe/codes/jszn/InternVL/models/Ours_Mini-InternVL-4B/models--OpenGVLab--Mini-InternVL-Chat-4B-V1-5/snapshots/6f97087daec17e4b033d4d846c0b64c09c4268cd" \
"/fuxi_team14/users/haoyanghe/codes/jszn/InternVL/models/Ours_Mini-InternVL-4B/models--OpenGVLab--Mini-InternVL-Chat-4B-V1-5/snapshots/6f97087daec17e4b033d4d846c0b64c09c4268cd/pointmae_pretrain.pth"