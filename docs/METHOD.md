# Method implementation

## Input representation

A source observation is a binary array of shape channels × time. All recordings
use the same channel map and sampling interval. An ACT transition starts activity
and an RTN transition ends it under the event-reconstruction convention. For the
TEP evaluation the selected window has 50 channels and 60 one-minute samples.
The array representation is not restricted to these dimensions, although the
provided experiment configurations are TEP-specific.

## Excerpt and full-episode input selection

The E variant selects a common excerpt length for each minibatch, then samples
source recordings and starts uniformly with replacement. Empty inputs remain
eligible. F uses the complete observation from each sampled source. Different
minibatches may have different lengths; lengths must be divisible by the patch
size. No flood selector is called during excerpt sampling.

## Encoder

Each non-overlapping three-sample patch contains values from all channels. Its
flattened Cq values are linearly projected to 48 dimensions. During pretraining,
30% of patch vectors (rounded, at least one) are replaced by a shared learned mask
vector. Sine–cosine positions are added, then two pre-normalized, bidirectional
Transformer layers with four heads, GELU, feed-forward width 96 and dropout 0.1
produce features followed by final layer normalization. Position vectors are
extended when more than the initially allocated number of patches is needed.

## Student and teacher

The reference input is the recorded observation. A student sees the event-corrupted
view with patch masking; a teacher processes the reference without corruption,
masking or dropout. The teacher starts as a parameter copy and is updated only by
an exponential moving average with coefficient 0.99. Teacher targets are evaluated
without gradients.

The predictor is Linear(48,96), GELU, Linear(96,48), followed by layer normalization
without learned affine parameters. The decoder is Linear(48,Cq). Pretraining uses:

1. Mean squared error between predicted and reference features at **all** patches.
2. Binary cross-entropy reconstruction of **all** reference cells, with equal total
   weight for active and inactive targets. Empty terms contribute zero.
3. A hinge penalty on the feature standard deviations of mean-pooled student
   outputs, using divisor B, stabilizer 1e-4, and target standard deviation 0.5.

The full objective is latent loss + 0.25 reconstruction loss + 0.10 variance loss.
MAPLE-E/F remove event corruption but retain masking and all loss terms. Recon-E
uses reconstruction alone. The auxiliary components are not used at inference.

## Supervised stage

A fresh linear classifier receives the concatenated mean and elementwise maximum
of the encoder's patch outputs (96 features). The encoder and classifier are
fine-tuned jointly with cross-entropy. Labels are available only in this stage.
All pretrained methods fine-tune on complete, unmasked, experimentally uncorrupted
observations. Scratch uses the same encoder and classifier without pretraining.
Scratch+Aug additionally samples from eight pre-generated event-corrupted views
per labelled recording. This bank is not refreshed during fine-tuning.

## Mapping to code and configurations

| Element | Implementation |
|---|---|
| Input loading, duplicate checks, nested label subsets | `src/afc_excerpt_ssl/data.py` |
| Patch encoder, classifier, EMA teacher | `src/afc_excerpt_ssl/models.py` |
| Registered paper names and ablations | `src/cora_maple/config.py` |
| Training objectives and optimizers | `src/cora_maple/learning.py` |
| ACT/RTN conversion, corruption and paired tests | `src/cora_maple/transforms.py` and `vendor/afc_robustness/` |
| CV source assignments and seed pairing | `src/cora_maple/protocol.py` |
| Checkpointed jobs and resumable orchestration | `src/cora_maple/runner.py` |
| Statistics, paired comparisons and LaTeX export | `src/cora_maple/report.py` |
| Explicit hyperparameters | `configs/base.yaml` |

AdamW uses learning rate 1e-3, weight decay 1e-4, betas (0.9,0.999), epsilon 1e-8,
and gradient-norm clipping at one. Pretraining and supervision have 800 and 400
updates respectively. Checkpoint selection uses the final update.
