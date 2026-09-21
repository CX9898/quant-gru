# Speech Commands GRU real-network baseline research

## Conclusion

The real-network comparison is based on Google Research `kws_streaming`'s GRU
keyword-spotting topology. The upstream model is a speech feature extractor
followed by one or more GRU layers, flatten, dropout, optional dense layers, and
a final classifier. The source is pinned to commit
`4700efb9afa54286b0e04473ba80a13e8461e25f` so later upstream changes do not
alter the test's provenance.

The repository test is intentionally a smaller **topology adaptation**:

```text
Speech Commands v0.02 waveform
  -> 20-coefficient MFCC sequence
  -> one standard GRU(input=20, hidden=64, return final state, stateless)
  -> dropout(p=0)
  -> 4-class dense classifier
```

It compares PyTorch FP32, native CUDA FP32, INT8 QAT, and INT16 QAT while only
replacing the recurrent operator. It is not a reproduction of Google's full
12-label recipe or its reported accuracy.

## Primary sources

Research date: 2026-09-21.

| Claim | Primary source |
|---|---|
| The official GRU model builds speech features, GRU layer(s), flatten, dropout, optional dense layers, and the final label classifier. | [Google Research `gru.py`, lines 68-110](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/models/gru.py#L68-L110) |
| The model interface supports returning only the last output and supports stateless execution. | [Google Research GRU flags, lines 25-65](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/models/gru.py#L25-L65) and [toy GRU parameters, lines 242-252](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/models/model_params.py#L242-L252) |
| Google's full 12-label GRU experiment uses a 400-unit stateless GRU, final output only, dropout 0.1, dense layers of 128 and 256 units, and SpecAugment. | [Google Research 12-label GRU experiment, lines 355-385](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/experiments/kws_experiments_paper_12_labels.md#L355-L385) |
| The official parser defaults to Speech Commands v0.02 and 16 kHz, 1-second clips. | [Google Research `base_parser.py`, lines 33-46](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/train/base_parser.py#L33-L46) and [lines 247-270](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/train/base_parser.py#L247-L270) |
| The official GRU recipe uses a 40 ms window, 20 ms stride, 40 mel bins, 20 DCT/MFCC coefficients, and a 7.6 kHz upper edge. | [Google Research 12-label GRU experiment, lines 362-369](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/experiments/kws_experiments_paper_12_labels.md#L362-L369) |
| The Google feature path computes a spectrogram, triangular mel filterbank, logarithm, and DCT. | [Google Research `input_data.py`, lines 478-502](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/data/input_data.py#L478-L502) |
| Google's loader removes the `_nohash_` suffix and hashes the remaining filename so related recordings remain in the same split. | [Google Research `input_data_utils.py`, lines 33-78](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/data/input_data_utils.py#L33-L78) |
| The v0.02 archive contains `testing_list.txt` and `validation_list.txt`; training is the WAV set minus both lists. | [TensorFlow Datasets Speech Commands builder, lines 150-169](https://github.com/tensorflow/datasets/blob/1401448b0c6c7aaf12bb5ee666a73fd6898650d1/tensorflow_datasets/datasets/speech_commands/speech_commands_dataset_builder.py#L150-L169) |
| The archive has SHA-256 `af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58` and size 2,428,923,189 bytes. | [TensorFlow Datasets checksum manifest](https://github.com/tensorflow/datasets/blob/1401448b0c6c7aaf12bb5ee666a73fd6898650d1/tensorflow_datasets/datasets/speech_commands/checksums.tsv) |

## Upstream model and test adaptation

The upstream defaults in `gru.py` are 400 GRU units, final output only,
stateful execution, dropout 0.1, and dense layers of 128 and 256 units. The
published 12-label experiment explicitly overrides `stateful` to `0`, retains
400 GRU units and both dense layers, and trains with SpecAugment. The smaller
test keeps the recurrent classification path but changes the following:

| Property | Full Google experiment | Repository test |
|---|---:|---:|
| labels | 10 commands plus unknown and silence | `yes`, `no`, `up`, `down` |
| GRU layers | 1 | 1 |
| hidden units | 400 | 64 |
| recurrent output | final output | final hidden state |
| stateful | no | no |
| dropout | 0.1 | 0 |
| intermediate dense layers | 128, 256 | none |
| final classifier | 12 classes | 4 classes |
| augmentation | resampling and SpecAugment | none |

The reduction keeps runtime suitable for an end-to-end CUDA regression and
isolates the recurrent implementation. Disabling dropout also prevents random
masks from amplifying small kernel differences. Because the capacity, labels,
augmentation, optimizer schedule, and training duration differ, the test's
accuracy must not be compared with Google's full recipe as an absolute
benchmark.

## Dataset split and deterministic subset

The test consumes the v0.02 archive's `validation_list.txt` and
`testing_list.txt` directly:

- validation contains paths in `validation_list.txt`;
- testing contains paths in `testing_list.txt`;
- training contains remaining WAV files after excluding both lists;
- only the four selected command directories are considered.

This follows the archive's official split membership. Google Research's own
loader derives membership with the stable SHA-1 rule in `which_set()` rather
than reading the lists. Both mechanisms preserve `_nohash_` speaker groups,
but the test deliberately treats the archive lists as authoritative so its
sample IDs are independently auditable.

Subsampling happens only after assigning the official split. For every split
and label, candidates are sorted, shuffled with a seed derived from the global
seed, split, and label index, and then truncated to the configured per-label
count. Therefore:

- each split is class-balanced;
- the selected paths are repeatable for the same seed and configuration;
- changing directory iteration order does not change the subset;
- no validation or test example participates in training or calibration.

The default comparison uses 128 training, 32 validation, and 32 testing
examples per label. It records a SHA-256 digest of the selected relative paths
for each split so repeated results can prove they used identical examples.

## Feature contract

The test uses the feature dimensions and frame parameters from the official
GRU experiment:

| Parameter | Value |
|---|---:|
| sample rate | 16,000 Hz |
| clip duration | 1,000 ms |
| analysis window | 40 ms / 640 samples |
| hop | 20 ms / 320 samples |
| mel bins | 40 |
| MFCC coefficients | 20 |
| sequence length | 49 frames |

The sequence length is `1 + floor((16000 - 640) / 320) = 49`, matching the
frame-count calculation in [Google Research `model_flags.py`, lines
37-56](https://github.com/google-research/google-research/blob/4700efb9afa54286b0e04473ba80a13e8461e25f/kws_streaming/models/model_flags.py#L37-L56).
The implementation uses `torchaudio.transforms.MFCC` with centered padding
disabled, HTK mel scaling, squared magnitude, and the experiment's 7.6 kHz
upper edge. It normalizes all splits with one scalar mean and standard
deviation computed from the selected training features only.

These choices preserve the official feature shape and main signal-processing
stages, but they are not a bit-exact reimplementation of TensorFlow's
`audio_spectrogram` and `audio_ops.mfcc`. Feature tensors are precomputed on
CPU and then transferred to CUDA by batch; that preprocessing is not a GRU CPU
fallback.

## Comparison contract

Four branches use the same selected examples, feature tensors, initial
parameters, batch order, loss, optimizer, learning rate, epoch count, and
gradient clipping:

| Branch | Recurrent operator | Purpose |
|---|---|---|
| baseline | `torch.nn.GRU` | PyTorch FP32 reference training |
| native CUDA FP32 | `QuantGRU(use_quantization=False)` | Isolate native operator behavior from quantization |
| INT8 QAT | `QuantGRU`, 8-bit quantization enabled | Measure 8-bit QAT behavior and logit error |
| INT16 QAT | `QuantGRU`, 16-bit quantization enabled | Measure bitwidth sensitivity and higher-precision behavior |

The baseline parameters are copied into every replacement branch before
training. QAT calibration uses a deterministic, class-balanced subset of the
training split and is refreshed from that same subset after each epoch before
validation. Quantized logit MAE is computed against the trained QAT branch's
own native-float path, on the same final parameters and test examples. This
isolates forward quantization error; it is not an MAE against a separately
trained PyTorch baseline.

All recurrent forward and backward work must execute on CUDA. CPU work is
limited to dataset I/O and feature preprocessing; there is no CPU recurrent
fallback in any `QuantGRU` branch.
