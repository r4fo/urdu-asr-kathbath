#!/usr/bin/env bash
# MFCC + GMM-HMM baseline with Kaldi (Member 3). Run from anywhere; it finds the repo root itself.
# Needs Kaldi (for example the Docker image kaldiasr/kaldi:latest, KALDI_ROOT=/opt/kaldi) on Linux or WSL2.
#
#   bash baselines/gmm_hmm/run.sh                  # train on train, decode DEV only, choose the LM weight on dev
#   bash baselines/gmm_hmm/run.sh --stage 4        # resume from a stage
#   bash baselines/gmm_hmm/run.sh --test           # decode TEST once with the dev-chosen settings (do this last)
#
# Stages: 0 data folders + lexicon + LM | 1 lang | 2 MFCC+CMVN | 3 monophone | 4 tri1 (deltas)
#         5 tri2 (LDA+MLLT) | 6 graph + decode dev + dev predictions | 7 (--test) decode test once
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$REPO/baselines/gmm_hmm"
W="$HERE/work"
KALDI_ROOT="${KALDI_ROOT:-/opt/kaldi}"
NJ_MAX="${NJ:-$(nproc)}"
stage=0
do_test=false
while [ $# -gt 0 ]; do
  case "$1" in
    --stage) stage="$2"; shift 2;;
    --test) do_test=true; shift;;
    *) echo "unknown option $1"; exit 1;;
  esac
done

mkdir -p "$W/conf"
cd "$W"
ln -sfn "$KALDI_ROOT/egs/wsj/s5/steps" steps
ln -sfn "$KALDI_ROOT/egs/wsj/s5/utils" utils
mkdir -p local && ln -sfn ../steps/score_kaldi.sh local/score.sh
cat > path.sh <<PATHSH
export KALDI_ROOT=$KALDI_ROOT
export PATH=\$PWD/utils:\$KALDI_ROOT/tools/openfst/bin:\$PWD:\$PATH
[ -f \$KALDI_ROOT/tools/config/common_path.sh ] && . \$KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
PATHSH
. ./path.sh
cat > conf/mfcc.conf <<'CONF'
--use-energy=false
--num-mel-bins=40
--num-ceps=13
--sample-frequency=16000
CONF
cmd=run.pl

# number of parallel jobs: at most one per speaker (Kaldi splits by speaker) and at most the CPU cores
njobs() { local n; n=$(wc -l < "data/$1/spk2utt"); echo $(( n < NJ_MAX ? n : NJ_MAX )); }

if $do_test; then stage=7; fi

if [ $stage -le 0 ]; then
  echo "== stage 0: data folders, lexicon, language model (train text only)"
  command -v flac >/dev/null || { [ "$(id -u)" = 0 ] && apt-get -qq update && apt-get -qq install -y flac; }
  python3 "$REPO/scripts/make_kaldi_dirs.py" --splits-dir "$REPO/data/splits" --out-dir data --repo-root "$REPO"
  python3 "$HERE/make_dict.py" --train-text data/train/text --dict-dir data/local/dict \
      --check data/dev/text $( [ -f data/test/text ] && echo data/test/text ) | tee data/oov_report.txt
  mkdir -p lm
  cut -d' ' -f2- data/train/text > lm/train_text.txt
  LMPLZ=$(command -v lmplz || true)
  if [ -z "$LMPLZ" ]; then
    if [ ! -x kenlm/build/bin/lmplz ]; then
      if command -v apt-get >/dev/null && [ "$(id -u)" = 0 ]; then
        apt-get -qq update && apt-get -qq install -y build-essential cmake libboost-program-options-dev \
          libboost-system-dev libboost-thread-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev >/dev/null
      fi
      [ -d kenlm ] || git clone -q https://github.com/kpu/kenlm.git
      mkdir -p kenlm/build && (cd kenlm/build && cmake .. >/dev/null && make -j4 lmplz >/dev/null)
    fi
    LMPLZ=kenlm/build/bin/lmplz
  fi
  "$LMPLZ" -o 3 --discount_fallback < lm/train_text.txt | gzip -c > lm/train_3gram.arpa.gz
  echo "LM: $(zcat lm/train_3gram.arpa.gz | head -8 | grep -E 'ngram [123]=' | tr '\n' ' ')"
fi

if [ $stage -le 1 ]; then
  echo "== stage 1: lang folders"
  utils/prepare_lang.sh --position-dependent-phones false data/local/dict "<unk>" data/local/lang data/lang
  utils/format_lm.sh data/lang lm/train_3gram.arpa.gz data/local/dict/lexicon.txt data/lang_test
fi

if [ $stage -le 2 ]; then
  echo "== stage 2: MFCC + CMVN"
  for x in train dev $( [ -f data/test/text ] && echo test ); do
    steps/make_mfcc.sh --nj "$(njobs $x)" --cmd "$cmd" --mfcc-config conf/mfcc.conf data/$x exp/make_mfcc/$x mfcc
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x mfcc
    utils/fix_data_dir.sh data/$x
  done
fi

if [ $stage -le 3 ]; then
  echo "== stage 3: monophone"
  steps/train_mono.sh --nj "$(njobs train)" --cmd "$cmd" data/train data/lang exp/mono
  steps/align_si.sh --nj "$(njobs train)" --cmd "$cmd" data/train data/lang exp/mono exp/mono_ali
fi

if [ $stage -le 4 ]; then
  echo "== stage 4: tri1 (MFCC + deltas)"
  steps/train_deltas.sh --cmd "$cmd" 2000 10000 data/train data/lang exp/mono_ali exp/tri1
  steps/align_si.sh --nj "$(njobs train)" --cmd "$cmd" data/train data/lang exp/tri1 exp/tri1_ali
fi

if [ $stage -le 5 ]; then
  echo "== stage 5: tri2 (LDA + MLLT)"
  steps/train_lda_mllt.sh --cmd "$cmd" --splice-opts "--left-context=3 --right-context=3" \
      2500 15000 data/train data/lang exp/tri1_ali exp/tri2
fi

if [ $stage -le 6 ] && ! $do_test; then
  echo "== stage 6: graph, decode DEV, choose the LM weight on dev"
  utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph
  steps/decode.sh --nj "$(njobs dev)" --cmd "$cmd" \
      --scoring-opts "--min_lmwt 7 --max_lmwt 17 --word_ins_penalty 0.0,0.5,1.0" \
      exp/tri2/graph data/dev exp/tri2/decode_dev
  best=$(cat exp/tri2/decode_dev/scoring_kaldi/best_wer)
  echo "$best"
  # best_wer ends with .../wer_<LMWT>_<penalty>
  echo "$best" | grep -o 'wer_[0-9]*_[0-9.]*' | sed 's/wer_//' | tr '_' ' ' > best_lmwt_penalty.txt
  read -r L P < best_lmwt_penalty.txt
  python3 "$HERE/to_predictions.py" --kaldi-text "exp/tri2/decode_dev/scoring_kaldi/penalty_$P/$L.txt" \
      --split "$REPO/data/splits/dev.csv" --out "$REPO/results/predictions/gmm_hmm_dev.csv"
  echo "Chosen on dev: LM weight $L, word insertion penalty $P. Now commit, then run with --test ONCE."
fi

if [ $stage -le 7 ] && $do_test; then
  echo "== stage 7: decode TEST once (settings are frozen)"
  [ -f best_lmwt_penalty.txt ] || { echo "run the dev stage first"; exit 1; }
  [ -f exp/tri2/decode_test/scoring_kaldi/best_wer ] && { echo "test was already decoded; not decoding again"; exit 1; }
  read -r L P < best_lmwt_penalty.txt
  start=$(date +%s.%N)
  steps/decode.sh --nj "$(njobs test)" --cmd "$cmd" \
      --scoring-opts "--min_lmwt $L --max_lmwt $L --word_ins_penalty $P" \
      exp/tri2/graph data/test exp/tri2/decode_test
  end=$(date +%s.%N)
  python3 "$HERE/to_predictions.py" --kaldi-text "exp/tri2/decode_test/scoring_kaldi/penalty_$P/$L.txt" \
      --split "$REPO/data/splits/test.csv" --out "$REPO/results/predictions/gmm_hmm.csv"
  rev=$(git -C "$KALDI_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
  python3 "$HERE/make_timing.py" --split "$REPO/data/splits/test.csv" \
      --decode-seconds "$(echo "$end - $start" | bc)" --jobs "$(njobs test)" \
      --lmwt "$L" --revision "$rev" --out "$REPO/results/timing/gmm_hmm.csv"
  echo "Next: python src/evaluate.py --split data/splits/test.csv --pred results/predictions/gmm_hmm.csv"
fi
