from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

pipe = pipeline(
    "./models/Mini-InternVL-Chat-4B-V1-5/models--OpenGVLab--Mini-InternVL-Chat-4B-V1-5/snapshots/6f97087daec17e4b033d4d846c0b64c09c4268cd",
    backend_config=TurbomindEngineConfig(session_len=8192),
)
image = load_image("./examples/image1.jpg")
response = pipe(("describe this image", image))
# print(response.text)

sess = pipe.chat(("describe this image", image))
sess = pipe.chat(
    "What is the panda doing?",
    session=sess,
)
print(sess.response.text)
# print(response.text)
