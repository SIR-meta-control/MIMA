# Contributing

[🏠 MIMA](README.md) · [📚 Documentation](docs/README.md)

Report a reproducible issue with the command, environment versions, model/data
identity and full error output. Keep hardware-specific configuration separate
from common defaults.

<a id="local-checks"></a>

## ✅ Local checks

```bash
python -m pip install -r requirements-dev.txt
python scripts/check_repository.py
ruff check .
ruff format --check .
```

The repository checker validates source syntax, documentation links, artifact
checksums and ROS package configuration. Full data hashing is available with
`python scripts/check_repository.py --verify-data`.

Validate behavior changes with the relevant inference, training or reproduction
command, and record the model, data and environment used.

<a id="research-artifacts"></a>

## 📦 Research artifacts

> [!IMPORTANT]
> Keep archived metric JSONs, selected checkpoints and source experiment records
> unchanged when changing runtime code. New experimental results should identify
> the checkpoint, partition, seed, evaluation protocol and environment. Update
> `MANIFEST.sha256` whenever a covered module file changes; update `data/SHA256SUMS`
> and the data inventory whenever a published data archive changes.

Do not commit extracted datasets, generated outputs, local service credentials
or machine-specific absolute paths. Preserve third-party copyright notices and
license texts. Read [Third-party notices](docs/THIRD_PARTY_NOTICES.md) before
changing vendored components.

<a id="pull-requests"></a>

## 🤝 Pull requests

Describe the concrete behavior change, relevant checks and any model/data
changes. Run the local checks above and include the command used for behavior
validation and its outcome.
Hardware-dependent changes need validation in the matching ROS and robot
environment.
