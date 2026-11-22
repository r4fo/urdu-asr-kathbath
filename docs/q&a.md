# Urdu Speech Recognition ” Technical Q&A

**Team 7 · Project 3**

Fifteen questions on our dataset, models and evaluation for ASR on the Kathbath Urdu corpus, with the answers we stand behind.

## Results at a glance

| System | Setting | WER |
|---|---|---:|
| **wav2vec 2.0** | **With KenLM** | **18.3%** |
| Whisper-small | Fine-tuned | 19.7% |
| Whisper-large-v3 | Zero-shot | 21.6% |
| wav2vec 2.0 | Without LM | 23.3% |

*Lower WER is better. Same normalization applied to every system.*

## Contents

1. [Dataset and methodology](#1-dataset-and-methodology)
2. [Models and architecture](#2-models-and-architecture)
3. [Evaluation and results](#3-evaluation-and-results)
4. [Limitations and future work](#4-limitations-and-future-work)

---

## 1. Dataset and methodology

### Q1. Why did we split the dataset by speaker instead of randomly by audio clip?

Random splitting could put the same speaker in both training and testing. The model might recognize their voice, microphone, or recording conditions, making test performance artificially high.

Speaker-disjoint splitting tests **generalization to unseen voices**.

### Q2. Why did we use only 25 hours instead of the full Kathbath dataset?

We used a controlled 25-hour subset and capped audio hours per speaker to prevent individual speakers from dominating the dataset. This also kept experiments manageable across multiple models.

### Q3. Why do we need three separate datasets: train, dev, and test?

- **Train:** used to learn model parameters.
- **Dev:** used to select checkpoints and tune hyperparameters.
- **Test:** used only after finalizing the system, to measure generalization.

Using the test set to tune a model would cause **data leakage**.

### Q4. Why did we remove only one recording during cleaning?

Kathbath already contains human-verified recordings. Our automated checks and manual inspections found very few problems. Removing additional recordings without evidence could unnecessarily reduce the dataset.

---

## 2. Models and architecture

### Q5. What is the difference between CTC, attention-based ASR, and RNN-T?

- **CTC:** predicts tokens from audio frames and collapses repeated tokens and blanks.
- **Attention:** uses a decoder that attends to relevant audio representations while generating text.
- **RNN-T:** combines acoustic and prediction networks, allowing streaming transcription.

All three were evaluated as course baselines.

### Q6. Why did pretrained models outperform models trained from scratch?

Pretrained models already learn useful speech representations from large datasets. Our training split contained only **17.6 hours**, which is relatively small for learning speech recognition from scratch. Fine-tuning builds on existing knowledge instead.

### Q7. Why did we fine-tune Whisper-small rather than Whisper-large-v3?

Whisper-small requires fewer computational resources and provides a practical fine-tuning target. We could then compare a smaller task-adapted model against a much larger zero-shot model.

### Q8. What is the difference between Whisper and wav2vec 2.0?

**Whisper** is an encoder-decoder speech recognition model trained to generate text from audio.

**wav2vec 2.0** is a self-supervised speech representation model. For ASR, it needs a trained transcription head, commonly using CTC.

### Q9. What does zero-shot mean in our experiment?

It means using an existing pretrained ASR checkpoint without additional training by our team on Kathbath Urdu.

> [!NOTE]
> The model was already trained by its original developers; zero-shot does not mean completely untrained.

---

## 3. Evaluation and results

### Q10. Why do we report both WER and CER?

WER measures word-level transcription errors, while CER measures character-level errors.

CER is particularly useful for Urdu because a single incorrect character can make an entire word count as incorrect under WER.

### Q11. How is Word Error Rate calculated?

$$
\text{WER} = \frac{\text{Substitutions} + \text{Deletions} + \text{Insertions}}{\text{Number of reference words}}
$$

For example, if a 100-word reference contains 10 substitutions, 3 deletions, and 2 insertions, WER is **15%**. Lower is better.

### Q12. Why is our fine-tuned Whisper-small better than Whisper-large-v3 zero-shot?

Our fine-tuned Whisper-small achieved **19.7% WER**, compared with **21.6%** for zero-shot Whisper-large-v3.

Fine-tuning adapted the smaller model to Kathbath Urdu speech and transcription patterns. This shows that task-specific adaptation can sometimes outperform a larger general-purpose model.

### Q13. Why did adding KenLM improve wav2vec 2.0?

The acoustic model predicts likely text from audio. KenLM adds information about which word sequences are linguistically probable.

Our wav2vec2 WER decreased from **23.3% to 18.3%** after adding the language model. Error analysis shows the main improvement came from fewer word substitutions.

### Q14. Why did we apply the same normalization to every model?

Different systems can produce different punctuation, Unicode letter variants, and number formats.

Applying one normalization policy to both references and predictions ensures that formatting differences do not unfairly influence the comparison.

### Q15. Can we conclude that wav2vec2 is 13 times faster than Whisper?

**Not as a controlled model-speed comparison.**

Our measured real-time factors were approximately **0.0043** for wav2vec2 with KenLM and **0.054** for fine-tuned Whisper-small. However, they were evaluated on different hardware with different decoding settings.

The observed ratio is approximately 13Ã—, but a fair speed benchmark would require identical hardware and comparable testing conditions.

---

## 4. Limitations and future work

Our evaluation uses **read speech from one corpus**, so performance may differ on spontaneous conversation, noisy recordings, regional accents, or specialized speech domains.

Future work could evaluate additional Urdu datasets, more varied speaking conditions, and language-model settings selected entirely through development-set experiments.
