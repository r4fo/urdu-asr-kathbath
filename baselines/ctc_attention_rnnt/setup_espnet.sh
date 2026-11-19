#!/usr/bin/env bash
# Install ESPnet (pinned) for the CTC / hybrid / RNN-T baselines.
# Works on Colab and on Linux/WSL2. About 5-10 minutes.
#
# Usage: bash baselines/ctc_attention_rnnt/setup_espnet.sh [ESPNET_ROOT]
#   ESPNET_ROOT defaults to $HOME/espnet. It holds the ESPnet recipe scripts
#   (egs2/TEMPLATE/asr1); the Python package is installed with pip.
set -euo pipefail

ESPNET_VERSION=202412           # pip version
ESPNET_TAG=v.202412             # matching git tag for the recipe scripts
ESPNET_ROOT=${1:-${ESPNET_ROOT:-$HOME/espnet}}

if [ ! -d "${ESPNET_ROOT}/egs2/TEMPLATE/asr1" ]; then
    git clone --depth 1 --branch "${ESPNET_TAG}" https://github.com/espnet/espnet.git "${ESPNET_ROOT}"
fi

# torch 2.4.1 is the version pinned in requirements.txt (wheels exist for Python 3.8-3.12).
if ! python3 -m pip install -q torch==2.4.1 torchaudio==2.4.1; then
    echo "WARNING: could not install torch 2.4.1 for $(python3 --version); keeping the installed torch."
    echo "Write the version printed below into the README section."
fi
python3 -m pip install -q "espnet==${ESPNET_VERSION}" tensorboard==2.18.0 matplotlib==3.9.2

# ESPnet's path.sh sources tools/activate_python.sh; point it at this python.
PYTHON_DIR=$(dirname "$(command -v python3)")
cat > "${ESPNET_ROOT}/tools/activate_python.sh" << EOF
#!/usr/bin/env bash
export PATH=${PYTHON_DIR}:\${PATH}
EOF

python3 -c "import espnet, torch; print('espnet', espnet.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
echo "ESPnet ready in ${ESPNET_ROOT}"
