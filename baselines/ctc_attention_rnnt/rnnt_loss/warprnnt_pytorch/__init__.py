"""Drop-in for ESPnet's `from warprnnt_pytorch import RNNTLoss`.

ESPnet's standard RNN-T model imports the warp-transducer package, which must
be compiled from source with CUDA. To avoid that on Colab, this module
provides the same class on top of torchaudio.functional.rnnt_loss, which
computes the same standard RNN-T loss (log-softmax over the joint network
output, then the forward-backward sum over all alignments). run.sh puts this
folder on PYTHONPATH only when no real warprnnt_pytorch is installed.
"""

import torch
import torchaudio


class RNNTLoss(torch.nn.Module):
    def __init__(self, blank=0, fastemit_lambda=0.0, reduction="mean"):
        super().__init__()
        self.blank = blank
        self.fastemit_lambda = fastemit_lambda
        self.reduction = reduction

    def forward(self, logits, targets, logit_lengths, target_lengths):
        # logits: (batch, T, U+1, vocab) raw joint-network output
        return torchaudio.functional.rnnt_loss(
            logits.float(),
            targets.int(),
            logit_lengths.int(),
            target_lengths.int(),
            blank=self.blank,
            reduction=self.reduction,
            fused_log_softmax=True,
            clamp=-1,
        )
