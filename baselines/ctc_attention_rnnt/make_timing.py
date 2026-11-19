"""Write results/timing/<system>.csv for one ESPnet baseline, in the team's
timing format (the same columns as models/whisper_finetune/transcribe.py):

  system,model,revision,num_beams,batch_size,hardware,audio_seconds,decode_seconds,rtf

The numbers come from ESPnet's own calculate_rtf.log for the test decoding:
decode_seconds is the sum of per-utterance decoding times (model loading is
not counted), audio_seconds the total audio length, rtf = decode / audio.
ESPnet decodes one utterance at a time (batch size 1), in one job.

Usage (run.sh calls it after decoding test):
  python baselines/ctc_attention_rnnt/make_timing.py --system ctc \
      --rtf-log <exp>/asr_ctc/decode_ctc/test/logdir/calculate_rtf.log \
      --decode-config baselines/ctc_attention_rnnt/conf/decode_ctc.yaml \
      --out results/timing/ctc.csv
"""

import argparse
import csv
import os
import re
import subprocess

FIELDS = ["system", "model", "revision", "num_beams", "batch_size",
          "hardware", "audio_seconds", "decode_seconds", "rtf"]
HERE = os.path.dirname(os.path.abspath(__file__))


def read_number(text, label):
    m = re.search(label + r":\s*([0-9.]+)", text)
    if not m:
        raise SystemExit(f"'{label}' not found in calculate_rtf.log")
    return float(m.group(1))


def hardware():
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.strip()
        if gpu:
            return gpu.splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        pass
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                return "CPU: " + line.split(":", 1)[1].strip()
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument("--rtf-log", required=True)
    parser.add_argument("--decode-config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    text = open(args.rtf_log, encoding="utf-8").read()
    audio = read_number(text, "Total audio duration")
    decode = read_number(text, "Total decoding time")
    beams = re.search(r"^beam_size:\s*(\d+)", open(args.decode_config).read(), re.M)
    revision = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerow({"system": args.system,
                    "model": f"ESPnet 202511 {args.system} (trained by us, conf/train_{args.system}.yaml)",
                    "revision": revision,
                    "num_beams": beams.group(1) if beams else "",
                    "batch_size": 1,
                    "hardware": hardware(),
                    "audio_seconds": round(audio, 2),
                    "decode_seconds": round(decode, 2),
                    "rtf": round(decode / audio, 4) if audio else ""})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
