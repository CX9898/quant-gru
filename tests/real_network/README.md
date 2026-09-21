# Speech Commands GRU real-network test

This opt-in test adapts Google Research `kws_streaming`'s GRU keyword-spotting
topology to the standard GRU interface supported by this repository:

```text
1 s / 16 kHz waveform
  -> 49 x 20 MFCC sequence
  -> one unidirectional GRU (hidden size 64, final state)
  -> dropout(0)
  -> four-class linear classifier
```

The upstream model is pinned to
[`google-research/google-research@4700efb`](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/models/gru.py#L70-L110).
Its standard path is speech features, one or more GRU layers, flatten, dropout,
optional dense layers, and the output classifier. The official toy GRU config
uses `return_sequences=0` and `stateful=0`; this test scales that standard cell
to hidden size 64 and omits the optional intermediate dense layers so that only
the recurrent implementation changes between branches.

This is a four-label real-data regression, not a reproduction of Google's full
12-label training recipe or reported accuracy.

## Comparison contract

All four branches use the same selected examples, precomputed MFCC tensors,
initial parameters, batch order, optimizer, learning rate, epochs, loss, and
gradient clipping:

- baseline: `torch.nn.GRU` on CUDA;
- native float: `QuantGRU(use_quantization=False)` using CUDA C++ kernels;
- INT8 QAT: `QuantGRU` with all quantization points set to 8 bit;
- INT16 QAT: `QuantGRU` with all quantization points set to 16 bit.

Both QAT branches use affine M+shift scale encoding and FP32 storage for integer
carrier values. Calibration uses the same deterministic, class-balanced 128
training examples before training and after every epoch. Validation and test
examples never participate in calibration.

The final quantization error compares each trained QAT model's quantized path
with that same model's native-float path. This isolates execution quantization
error from differences caused by separately trained model parameters.

## Run

Build and install the CUDA library and Python extension first, then run:

```bash
tests/real_network/run_speech_commands_gru_test.sh \
  --dataset-root /path/to/speech_commands_v0.02
```

The generated report is written to
`tests/results/speech_commands_gru_training.json`, which is intentionally
ignored by Git.

## Acceptance thresholds

The test requires:

- baseline and native-float validation accuracy >= 70% and test accuracy >= 80%;
- native float no more than 1 percentage point below the PyTorch baseline;
- both QAT branches validation accuracy >= 70% and test accuracy >= 80%;
- each QAT branch no more than 5 points below baseline validation accuracy;
- INT8 logit MAE < 0.03, normalized MAE < 1.5%, cosine >= 0.9998, and prediction
  agreement with its own float path >= 98%;
- INT16 logit MAE and normalized MAE < 0.001, cosine >= 0.999999, and exact
  prediction agreement with its own float path;
- INT16 logit MAE must be less than 5% of INT8 logit MAE;
- every initial and refreshed calibration subset must contain exactly 32
  examples from each label.

Representative RTX 6000D results with seed `20260921` are:

| Metric | `torch.nn.GRU` | Native FP32 | INT8 QAT | INT16 QAT |
|---|---:|---:|---:|---:|
| Final train loss | 0.55004 | 0.55011 | 0.50252 | 0.51720 |
| Best validation accuracy | 74.22% | 74.22% | 78.91% | 76.56% |
| Final test accuracy | 83.59% | 83.59% | 87.50% | 82.81% |
| Logit MAE vs own float path | N/A | N/A | 0.02046 | 0.000243 |
| Normalized logit MAE | N/A | N/A | 1.081% | 0.0132% |
| Logit cosine | N/A | N/A | 0.999884 | 0.999999988 |
| Quantized/float prediction agreement | N/A | N/A | 100% | 100% |

The INT8 and INT16 branches are trained independently, so their classification
accuracy need not be monotonic with bitwidth. The same-model quantized-versus-
float logit comparison is the direct bitwidth precision check.
