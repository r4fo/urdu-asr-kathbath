"""Download a speaker-balanced subset of Kathbath Urdu into data/raw/.

Source: ai4bharat/Kathbath on Hugging Face, config "urdu" (splits train + valid).
The official valid speakers all appear in train, so we pool both splits and
let scripts/split.py make speaker-disjoint train/dev/test splits later.

Selection: we cap the hours per speaker so the total is about --target-hours.
Speakers with less data than the cap keep all their utterances; speakers with
more are sampled down with a fixed seed. This keeps all speakers and stops a
few speakers with many hours from dominating the data.

The dataset stores audio as 24-bit FLAC. We save it as 16-bit FLAC (same 16 kHz
sample rate, no other change), which halves the size so the data fits in the repo.

Output:
  data/raw/audio/<speaker_id>/<utt_id>.flac   16 kHz, 16-bit FLAC
  data/raw/metadata.csv                        utt_id,path,speaker_id,gender,duration,text,source_split

Usage (repo root, venv active, logged in to Hugging Face):
  python scripts/download_kathbath_urdu.py --target-hours 25 --seed 42
  python scripts/download_kathbath_urdu.py --reencode-existing   # convert files already downloaded
"""

import argparse
import io
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfFileSystem, list_repo_files

REPO = "ai4bharat/Kathbath"
META_COLUMNS = ["fname", "speaker_id", "gender", "duration", "text"]
AUDIO_COLUMN = "audio_filepath"
RAW_DIR = Path("data/raw")


def urdu_parquet_files():
    files = [f for f in list_repo_files(REPO, repo_type="dataset")
             if f.endswith(".parquet") and "urdu" in f.lower()]
    if not files:
        raise SystemExit("No Urdu parquet files found in the dataset repo.")
    return sorted(files)


def split_name(path):
    p = path.lower()
    return "valid" if "valid" in p else "train" if "train" in p else "other"


def read_metadata(fs, files):
    frames = []
    for path in files:
        with fs.open(f"datasets/{REPO}/{path}", "rb") as fh:
            df = pq.read_table(fh, columns=META_COLUMNS).to_pandas()
        df["source_split"] = split_name(path)
        df["parquet_file"] = path
        frames.append(df)
    meta = pd.concat(frames, ignore_index=True)
    meta["utt_id"] = meta["fname"].str.rsplit(".", n=1).str[0]
    if meta["utt_id"].duplicated().any():
        raise SystemExit("Duplicate utt_id values found; check the fname column.")
    return meta


def find_cap(hours_per_speaker, target_hours):
    """Largest per-speaker cap (hours) whose capped total does not exceed the target."""
    if hours_per_speaker.sum() <= target_hours:
        return float(hours_per_speaker.max())
    lo, hi = 0.0, float(hours_per_speaker.max())
    for _ in range(60):
        mid = (lo + hi) / 2
        if hours_per_speaker.clip(upper=mid).sum() > target_hours:
            hi = mid
        else:
            lo = mid
    return lo


def select_subset(meta, target_hours, seed):
    hours = meta.groupby("speaker_id")["duration"].sum() / 3600
    cap_h = find_cap(hours, target_hours)
    chosen = []
    for spk, group in meta.groupby("speaker_id"):
        group = group.sample(frac=1.0, random_state=seed)  # fixed-seed order
        keep = group[group["duration"].cumsum() <= cap_h * 3600]
        if keep.empty:  # always keep at least one utterance per speaker
            keep = group.iloc[:1]
        chosen.append(keep)
    subset = pd.concat(chosen).sort_values(["speaker_id", "utt_id"]).reset_index(drop=True)
    return subset, cap_h


def save_flac16(audio, sr, target):
    """Write mono audio as 16-bit FLAC at its original sample rate."""
    sf.write(target, audio, sr, format="FLAC", subtype="PCM_16")


def reencode_existing():
    meta = pd.read_csv(RAW_DIR / "metadata.csv", encoding="utf-8")
    done = 0
    for path in meta["path"]:
        if sf.info(path).subtype == "PCM_16":
            continue
        audio, sr = sf.read(path, dtype="float32")
        save_flac16(audio, sr, path)
        done += 1
        if done % 2000 == 0:
            print(f"  re-encoded {done}")
    print(f"Re-encoded {done} files to 16-bit FLAC ({len(meta) - done} were already 16-bit).")


def write_audio(fs, subset):
    wanted = set(subset["fname"])
    out_paths = {}
    for path in sorted(subset["parquet_file"].unique()):
        with fs.open(f"datasets/{REPO}/{path}", "rb") as fh:
            pf = pq.ParquetFile(fh)
            for batch in pf.iter_batches(columns=["fname", "speaker_id", AUDIO_COLUMN], batch_size=256):
                rows = batch.to_pylist()
                for row in rows:
                    if row["fname"] not in wanted:
                        continue
                    utt_id = row["fname"].rsplit(".", 1)[0]
                    target = RAW_DIR / "audio" / str(row["speaker_id"]) / f"{utt_id}.flac"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    audio, sr = sf.read(io.BytesIO(row[AUDIO_COLUMN]["bytes"]), dtype="float32")
                    save_flac16(audio, sr, target)
                    out_paths[row["fname"]] = target.as_posix()
        print(f"  done {path}: {len(out_paths)} files so far")
    return out_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-hours", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reencode-existing", action="store_true",
                        help="only convert already downloaded files to 16-bit FLAC")
    args = parser.parse_args()
    if args.reencode_existing:
        reencode_existing()
        return

    fs = HfFileSystem()
    files = urdu_parquet_files()
    print(f"Reading metadata from {len(files)} parquet files...")
    meta = read_metadata(fs, files)
    subset, cap_h = select_subset(meta, args.target_hours, args.seed)
    print(f"Cap per speaker: {cap_h:.2f} h -> {len(subset)} utterances, "
          f"{subset['duration'].sum() / 3600:.2f} h, {subset['speaker_id'].nunique()} speakers")

    print("Downloading audio for the selected utterances...")
    paths = write_audio(fs, subset)
    missing = set(subset["fname"]) - set(paths)
    if missing:
        raise SystemExit(f"{len(missing)} selected utterances were not found in the parquet files.")

    subset["path"] = subset["fname"].map(paths)
    cols = ["utt_id", "path", "speaker_id", "gender", "duration", "text", "source_split"]
    subset[cols].to_csv(RAW_DIR / "metadata.csv", index=False, encoding="utf-8")

    stats = subset.groupby("gender").agg(speakers=("speaker_id", "nunique"),
                                         utterances=("utt_id", "count"),
                                         hours=("duration", lambda d: d.sum() / 3600))
    print(stats.round(2))
    print(f"Wrote {RAW_DIR / 'metadata.csv'}")


if __name__ == "__main__":
    main()
