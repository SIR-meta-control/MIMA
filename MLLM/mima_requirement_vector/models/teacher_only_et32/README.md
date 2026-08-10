# Teacher-only ET32 Artifacts

This directory records the identity of the MLLM-distilled model used in the
reported experiment. The expected `student.joblib`, `metadata.json`, 85-D
feature extractor, and local service implementation are not included in the
available release assets.

Do not place a different model under these filenames unless its SHA-256 agrees
with `checksums.json`. Until the expected files are supplied, use the service
client in this repository with a verified compatible endpoint and pass
`--expected-model strict_student_teacher_only_et32_s32768`.
