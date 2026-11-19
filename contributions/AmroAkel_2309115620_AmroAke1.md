# Amro Akel

- Student ID: 2309115620
- GitHub: AmroAke1
- Role: Member 4, neural baselines we train: CTC, hybrid CTC/attention, RNN-T (ESPnet)

## What I did

- Set up ESPnet 202511 on Google Colab (`baselines/ctc_attention_rnnt/setup_espnet.sh`, `colab.ipynb`)
- Converted the shared split files into ESPnet data folders (`prepare_data.py`) and ESPnet output back into prediction files (`to_predictions.py`), with tests (`tests/test_ctc_attention_rnnt.py`)
- Wrote three configs with the same small Conformer encoder: CTC, hybrid CTC/attention (CTC weight 0.3), RNN-T (`baselines/ctc_attention_rnnt/conf/`)
- Tested the whole pipeline (data, training, decoding, prediction file) for all three systems on a 50-clip sample
