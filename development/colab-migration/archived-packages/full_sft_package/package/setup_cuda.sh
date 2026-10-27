#!/usr/bin/env bash
set -euo pipefail
PACKAGE=/content/lingbot-stage2/package
python -m pip install uv
python -m uv python install 3.12
if [ ! -x /content/lingbot-venv/bin/python ]; then
  python -m uv venv --python 3.12 /content/lingbot-venv
fi
PY=/content/lingbot-venv/bin/python
export PYTHONNOUSERSITE=1
python -m uv pip install --python "$PY" -r "$PACKAGE/lingbot/requirements.txt"
python -m uv pip install --python "$PY" setuptools wheel packaging ninja
MAX_JOBS=4 python -m uv pip install --python "$PY" --no-build-isolation flash-attn==2.8.3
python -m uv pip install --python "$PY" --no-deps 'lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz'
python -m uv pip install --python "$PY" -e "$PACKAGE/lingbot" --no-deps
python -m uv pip install --python "$PY" -r "$PACKAGE/lingbot/requirements-depth.txt"
python -m uv pip install --python "$PY" -r "$PACKAGE/lingbot/requirements.txt"
python -m uv pip install --python "$PY" numpydantic==1.9.0 --no-deps
python -m uv pip install --python "$PY" --no-deps 'utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183' 'pipeline @ git+https://github.com/EasternJournalist/pipeline.git@866f059d2a05cde05e4a52211ec5051fd5f276d6'
python -m uv pip install --python "$PY" -e "$PACKAGE/lingbot/lingbotvla/models/vla/vision_models/lingbot-depth" --no-deps
python -m uv pip install --python "$PY" -e "$PACKAGE/lingbot/lingbotvla/models/vla/vision_models/MoGe" --no-deps
"$PY" -c "import torch, flash_attn; assert torch.__version__.split('+')[0]=='2.8.0'; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, flash_attn.__version__)"
