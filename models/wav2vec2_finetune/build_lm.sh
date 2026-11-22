#!/usr/bin/env bash
# Build a 3-gram KenLM language model from the TRAIN split text only (never dev or test,
# so the LM cannot have seen test sentences). Output: lm/train_3gram.arpa
#
# Run on Linux, WSL or Colab, from the repo root:   bash models/wav2vec2_finetune/build_lm.sh
# It builds KenLM's lmplz once if it is not installed (about 5 minutes on Colab).
set -euo pipefail
export CMAKE_POLICY_VERSION_MINIMUM=3.5  # KenLM asks for CMake < 3.5, which CMake 4 refuses otherwise
mkdir -p lm

python - <<'PY'
import pandas as pd
text = pd.read_csv("data/splits/train.csv", encoding="utf-8")["text"]
lines = [" ".join(str(t).split()) for t in text]
open("lm/train_text.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"lm/train_text.txt: {len(lines)} sentences from the train split")
PY

LMPLZ=$(command -v lmplz || true)
if [ -z "$LMPLZ" ]; then
  if [ ! -x kenlm/build/bin/lmplz ]; then
    if command -v apt-get >/dev/null && [ "$(id -u)" = 0 ]; then
      apt-get -qq update && apt-get -qq install -y build-essential cmake libboost-program-options-dev libboost-system-dev \
        libboost-thread-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev >/dev/null
    fi
    [ -d kenlm ] || git clone -q https://github.com/kpu/kenlm.git
    mkdir -p kenlm/build && (cd kenlm/build && cmake .. >/dev/null && make -j4 lmplz build_binary >/dev/null)
  fi
  LMPLZ=kenlm/build/bin/lmplz
fi

"$LMPLZ" -o 3 --discount_fallback < lm/train_text.txt > lm/train_3gram.arpa
echo "Wrote lm/train_3gram.arpa ($(du -h lm/train_3gram.arpa | cut -f1))"
