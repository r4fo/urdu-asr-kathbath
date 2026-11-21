"""Write the timing row of the GMM-HMM system in the team format (results/timing/gmm_hmm.csv).

Columns: system,model,revision,num_beams,batch_size,hardware,audio_seconds,decode_seconds,rtf
audio_seconds is the total duration of the test split; decode_seconds is the wall time of Kaldi's
steps/decode.sh on it (lattice decoding with beam 13, run with --jobs parallel jobs on this machine's CPU).
"""
import argparse
import csv
import os
import platform


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", required=True)
    parser.add_argument("--decode-seconds", type=float, required=True)
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument("--lmwt", required=True)
    parser.add_argument("--revision", default="unknown", help="Kaldi git revision")
    parser.add_argument("--hardware", default=None, help="default: CPU name and core count")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.split, encoding="utf-8", newline="") as f:
        audio = sum(float(r["duration"]) for r in csv.DictReader(f))
    hardware = args.hardware or f"CPU only, {cpu_name()}, {os.cpu_count()} cores"
    model = (f"Kaldi MFCC + GMM-HMM tri2 (LDA+MLLT), grapheme lexicon, 3-gram LM, LM weight {args.lmwt}, "
             f"{args.jobs} parallel jobs (trained by us, baselines/gmm_hmm/run.sh)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "model", "revision", "num_beams", "batch_size", "hardware",
                    "audio_seconds", "decode_seconds", "rtf"])
        w.writerow(["gmm_hmm", model, args.revision, 13, 1, hardware, f"{audio:.2f}",
                    f"{args.decode_seconds:.2f}", f"{args.decode_seconds / audio:.4f}"])
    print(f"wrote {args.out} (RTF {args.decode_seconds / audio:.4f})")


if __name__ == "__main__":
    main()
