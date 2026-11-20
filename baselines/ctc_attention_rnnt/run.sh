#!/usr/bin/env bash
# Train, decode and export one ESPnet baseline: ctc, hybrid or rnnt.
#
# Usage (from the repository root):
#   bash baselines/ctc_attention_rnnt/run.sh <ctc|hybrid|rnnt> [options]
#
# Options:
#   --splits-dir DIR   split CSVs (default data/splits)
#   --name NAME        name of this run's work folder (default full; use sample for the 50-clip test)
#   --expdir DIR       where checkpoints go (default work/<name>/exp). On Colab, point it at
#                      Google Drive so training resumes after a disconnect.
#   --stage N          first asr.sh stage (default 2). 10 = stats, 11 = train, 12 = decode.
#   --stop-stage N     last asr.sh stage (default 12). Stage 13 (ESPnet scoring) is not used:
#                      it needs sclite, and all official scores come from src/evaluate.py.
#   --test             also decode test. Do this ONCE, after all settings are frozen on dev.
#   --ngpu N           GPUs for training (default 1; 0 = CPU)
#   --asr-args "..."   extra training arguments, e.g. "--max_epoch 1"
#
# Environment: ESPNET_ROOT (default $HOME/espnet), set up by setup_espnet.sh.
#
# Output:
#   dev:  results/predictions/<system>_dev.csv       (--name full; other names: work/<name>/predictions_dev/)
#   test: results/predictions/<system>.csv           (only with --test and --name full;
#                                                    other names write work/<name>/predictions_test/)
#   results/timing/<system>.csv                      (test decoding RTF, team format; with --test and --name full)
#   work/<name>/timing_<system>.txt                  (notes: hardware, training and decoding time)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${HERE}/../.." && pwd)
ESPNET_ROOT=${ESPNET_ROOT:-$HOME/espnet}

system=${1:?usage: run.sh <ctc|hybrid|rnnt> [options]}
shift
case "${system}" in ctc|hybrid|rnnt) ;; *) echo "unknown system: ${system}"; exit 1 ;; esac

splits_dir=${REPO_ROOT}/data/splits
name=full
expdir=
stage=2
stop_stage=12
do_test=false
ngpu=1
asr_args=
while [ $# -gt 0 ]; do
    case "$1" in
        --splits-dir) splits_dir=$(cd "$2" && pwd); shift 2 ;;
        --name) name=$2; shift 2 ;;
        --expdir) mkdir -p "$2"; expdir=$(cd "$2" && pwd); shift 2 ;;
        --stage) stage=$2; shift 2 ;;
        --stop-stage) stop_stage=$2; shift 2 ;;
        --test) do_test=true; shift ;;
        --ngpu) ngpu=$2; shift 2 ;;
        --asr-args) asr_args=$2; shift 2 ;;
        *) echo "unknown option: $1"; exit 1 ;;
    esac
done

work=${HERE}/work/${name}
expdir=${expdir:-${work}/exp}
test_sets=dev
${do_test} && test_sets="dev test"
mkdir -p "${work}" "${expdir}"

# 1. Data folders from the split CSVs (also checks that no speaker is in two splits).
python3 "${HERE}/prepare_data.py" --splits-dir "${splits_dir}" --out-dir "${work}/data"

# 2. ESPnet recipe folder inside the ESPnet tree (its path.sh expects egs2/<name>/asr1).
recipe=${ESPNET_ROOT}/egs2/urdu_kathbath/asr1
if [ ! -e "${recipe}/asr.sh" ]; then
    "${ESPNET_ROOT}/egs2/TEMPLATE/asr1/setup.sh" "${recipe}"
fi
cp "${HERE}"/conf/*.yaml "${recipe}/conf/"
ln -sfn "${work}/data" "${recipe}/data"

# RNN-T loss: use the torchaudio-based drop-in unless warp-transducer is installed.
if [ "${system}" = rnnt ] && ! python3 -c "import warprnnt_pytorch" 2>/dev/null; then
    export PYTHONPATH="${HERE}/rnnt_loss${PYTHONPATH:+:${PYTHONPATH}}"
fi

# ESPnet saves training plots with matplotlib. Colab sets MPLBACKEND to its notebook
# backend, which ESPnet's environment does not have, so use the file-only backend.
export MPLBACKEND=Agg

gpu_inference=true
[ "${ngpu}" -eq 0 ] && gpu_inference=false

# The character token list lives next to the checkpoints, so a Colab disconnect
# (which wipes work/ but not Google Drive) cannot change it.
mkdir -p "${expdir}/ur_token_list"
ln -sfn "${expdir}/ur_token_list" "${work}/data/ur_token_list"

asr() {  # asr <first stage> <last stage>
    (
        cd "${recipe}"
        ./asr.sh \
            --stage "$1" --stop_stage "$2" \
            --lang ur --fs 16k --audio_format flac --feats_type raw \
            --token_type char \
            --use_lm false \
            --ngpu "${ngpu}" --nj 4 --inference_nj 1 --gpu_inference "${gpu_inference}" \
            --max_wav_duration 30 \
            --dumpdir "${work}/dump" --expdir "${expdir}" \
            --train_set train --valid_set dev --test_sets "${test_sets}" \
            --asr_config "conf/train_${system}.yaml" \
            --inference_config "conf/decode_${system}.yaml" \
            --inference_asr_model valid.loss.best.pth \
            --asr_tag "${system}" --inference_tag "decode_${system}" \
            --bpe_train_text data/train/text --lm_train_text data/train/text \
            --asr_args "${asr_args}"
    )
}

# 3. ESPnet stages 2-12: format audio (2-4), character token list (5), stats (10),
#    train (11), decode (12). Stages 6-9 are language-model stages and are skipped.
start=$(date +%s)
# The formatted audio (work/<name>/dump) is local. If it is missing, for example after
# a Colab disconnect or when test is decoded for the first time, rebuild it first.
if [ "${stage}" -gt 4 ]; then
    for dset in train dev ${test_sets}; do
        if [ ! -s "${work}/dump/raw/${dset}/wav.scp" ]; then
            echo "dump for ${dset} missing, rebuilding stages 2-4"
            asr 2 4
            break
        fi
    done
fi
# The token list is built once per --expdir (stage 5) and never rebuilt, so it always
# matches the checkpoints in that folder.
if [ "${stage}" -gt 5 ] && [ ! -s "${expdir}/ur_token_list/char/tokens.txt" ]; then
    echo "token list missing in ${expdir}, building it (stage 5)"
    asr 5 5
fi
asr "${stage}" "${stop_stage}"
end=$(date +%s)

# 4. Prediction CSVs in the shared utt_id,hyp format.
#    ESPnet decodes the valid set (dev) into decode_<system>/org/dev, other sets into decode_<system>/<set>.
asr_exp=${expdir}/asr_${system}
decode_dir() { if [ "$1" = dev ]; then echo "${asr_exp}/decode_${system}/org/dev"; else echo "${asr_exp}/decode_${system}/$1"; fi; }
for dset in ${test_sets}; do
    hyp=$(decode_dir "${dset}")/text
    [ -f "${hyp}" ] || continue
    # Only the real run (--name full) writes into results/predictions/
    # (test: <system>.csv, dev: <system>_dev.csv, as the other systems do).
    if [ "${name}" = full ] && [ "${dset}" = test ]; then
        out=${REPO_ROOT}/results/predictions/${system}.csv
    elif [ "${name}" = full ]; then
        out=${REPO_ROOT}/results/predictions/${system}_${dset}.csv
    else
        out=${work}/predictions_${dset}/${system}.csv
    fi
    python3 "${HERE}/to_predictions.py" --espnet-text "${hyp}" --split "${splits_dir}/${dset}.csv" --out "${out}"
done

# 5. Timing. The test RTF goes to results/timing/<system>.csv in the team format
#    (one job, so per-utterance times are not slowed by a second job on the GPU).
#    Training time and dev RTF go to a notes file for the README.
if ${do_test} && [ "${name}" = full ]; then
    python3 "${HERE}/make_timing.py" --system "${system}" \
        --rtf-log "$(decode_dir test)/logdir/calculate_rtf.log" \
        --decode-config "${HERE}/conf/decode_${system}.yaml" \
        --out "${REPO_ROOT}/results/timing/${system}.csv"
fi
{
    echo "== ${system} $(date '+%Y-%m-%d %H:%M') stages ${stage}-${stop_stage}, wall time $(( end - start )) s"
    echo "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
    if [ -f "${asr_exp}/train.log" ]; then
        echo "train: $(grep -o 'elapsed time [0-9]* seconds' "${asr_exp}/train.log" | tail -1)"
    fi
    for dset in ${test_sets}; do
        rtf=$(decode_dir "${dset}")/logdir/calculate_rtf.log
        if [ -f "${rtf}" ]; then
            echo "decode ${dset}: $(grep -E 'RTF|Total decoding time' "${rtf}" | tr '\n' ' ')"
        fi
    done
} | tee -a "${work}/timing_${system}.txt"
