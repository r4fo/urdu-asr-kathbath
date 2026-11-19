#!/usr/bin/env bash
# Install ESPnet (pinned) for the CTC / hybrid / RNN-T baselines.
# Works on Colab and on Linux/WSL2. About 10 minutes (Colab: about 5).
#
# Usage: bash baselines/ctc_attention_rnnt/setup_espnet.sh [ESPNET_ROOT]
#   ESPNET_ROOT defaults to $HOME/espnet. It holds the ESPnet recipe scripts
#   (egs2/TEMPLATE/asr1) and, in tools/venv, a Python 3.12 environment with
#   ESPnet and torch. A separate environment is used because Colab's Python
#   (3.13) has no wheels for torch 2.4.1 or ESPnet's sentencepiece; uv
#   downloads Python 3.12 by itself. ESPnet's path.sh activates it, so
#   run.sh needs nothing else.
set -euo pipefail

ESPNET_VERSION=202511           # pip version (202412 needs numpy<1.24, no wheel for Python 3.12+)
ESPNET_TAG=v.202511             # matching git tag for the recipe scripts
PYTHON_VERSION=3.12
UV_VERSION=0.11.32
ESPNET_ROOT=${1:-${ESPNET_ROOT:-$HOME/espnet}}

# Re-clone if an older ESPnet version is already there.
if [ -d "${ESPNET_ROOT}" ] && [ "$(cat "${ESPNET_ROOT}/.urdu_asr_tag" 2>/dev/null)" != "${ESPNET_TAG}" ]; then
    echo "Removing ${ESPNET_ROOT} (not ${ESPNET_TAG})"
    rm -rf "${ESPNET_ROOT}"
fi
if [ ! -d "${ESPNET_ROOT}/egs2/TEMPLATE/asr1" ]; then
    git clone -q --depth 1 --branch "${ESPNET_TAG}" https://github.com/espnet/espnet.git "${ESPNET_ROOT}"
    echo "${ESPNET_TAG}" > "${ESPNET_ROOT}/.urdu_asr_tag"
fi

command -v uv > /dev/null || python3 -m pip install -q "uv==${UV_VERSION}"
VENV=${ESPNET_ROOT}/tools/venv
[ -x "${VENV}/bin/python" ] || uv venv -q --python "${PYTHON_VERSION}" "${VENV}"

# torch 2.4.1 is the version pinned in requirements.txt.
uv pip install -q --python "${VENV}/bin/python" \
    torch==2.4.1 torchaudio==2.4.1 "espnet==${ESPNET_VERSION}" \
    tensorboard==2.18.0 matplotlib==3.9.2 resampy==0.4.3

# ESPnet's path.sh sources tools/activate_python.sh; point it at the environment.
cat > "${ESPNET_ROOT}/tools/activate_python.sh" << EOF
#!/usr/bin/env bash
export PATH=${VENV}/bin:\${PATH}
export MPLBACKEND=Agg
EOF

"${VENV}/bin/python" -c "import sys, espnet, torch; print('python', sys.version.split()[0], '| espnet', espnet.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
echo "ESPnet ready in ${ESPNET_ROOT}"
