# Method and controlled ablations

## Representation and architecture

An input X has shape C × T, with binary active/inactive alarm states on a fixed
sample grid. Alarm-channel identity/order and sampling interval are consistent
between pretraining and downstream use. Each patch contains q=3 consecutive samples
from every channel. A linear Cq→96 map and sinusoidal relative-to-input positions
feed a bidirectional three-layer, four-head pre-normalized Transformer (FFN 192,
GELU, dropout 0.1), followed by LayerNorm. Mean and elementwise maximum pooling
are concatenated into 192 features and passed to a new linear softmax classifier.
For C=50, K=5 the encoder has 239,136 parameters and the classifier 240,101.

E uniformly samples one length per minibatch from {15,30,45,60}; each member samples
a source with replacement and a valid start uniformly. F uses complete supplied
60-sample episodes. E requires no flood labels or boundary detector, but the current
benchmark supplies only curated fault episodes. All input samples are retained.

Validity masks identify unavailable padded samples. Attention ignores fully padded
patches, a partial final patch has its missing cells set to zero, mean pooling is
weighted by real samples/patch width, and maximum pooling excludes entirely padded
patches. Reconstruction and latent losses exclude missing cells/patches. Input
length can vary across batches and, where needed, within batches. Padding is never
interpreted as a real return-to-normal interval.

## Self-supervised learning

The teacher begins as a copy of the student, with no gradient updates, masking,
corruption or dropout. The student sees event/state-corrupted and masked views
(mask ratio 0.3, rounded with a minimum of one patch), independently of whether
an unchanged input was sampled. A 96→192→96 GELU predictor maps student features
to teacher targets; its output has affine-free LayerNorm.

The loss is mean squared latent-prediction error over all real patch positions,
plus 0.50 balanced binary cross-entropy reconstruction and 0.10 variance penalty.
Active/inactive targets each receive half of the reconstruction weight when both
are present; an empty class term contributes zero. The population standard
deviation of each mean-pooled feature is sqrt(var + 1e-4), penalized below 0.5.
The teacher follows EMA coefficient 0.995 after every optimizer update. No full
VICReg covariance penalty or KL consistency term is used.

Corrupted pretraining samples false_full, missing_full or timing uniformly, with
20% identity views and severity uniformly 0.05–0.30. Delay is absent because a
cropped prefix would destroy student/teacher patch-coordinate alignment. Uncorrupted
pretraining uses the same masking, losses, architecture and matched update
schedule without experimental corruption. Its source pool is identical to its
corrupted counterpart.

## Classifier training

Every method minimizes clean cross-entropy on the same labelled original episodes:

    L_cls = CE(classifier(X), y).

The student encoder and newly initialized linear classification head are updated
jointly. No supervised corruption, patch masking, extra labelled examples,
pseudo-labels, or consistency penalty are used. Dropout remains enabled during
training and disabled at inference. 

At inference, exactly one class is returned per supplied observation. Delayed test
windows are shorter real observations, not full observations padded with trustworthy
zeros. Delay is never sampled in either training stage under this release.

## Optimizer and fixed schedules

| Setting | E | F |
|---|---:|---:|
| SSL updates | 4,000 | 4,000 |
| Supervised updates | 2,400 | 2,400 |
| SSL batch | 32 | 32 |
| Supervised batch | min(32,kK) | min(32,kK) |
| SSL peak LR / weight decay | 5e-4 / 5e-4 | Same |
| Fine-tuning encoder/head peak LR | 3.5e-4 / 1e-3 | Same |
| Fine-tuning weight decay | 0.01 | Same |
| Warmup SSL / supervised | 80 / 50 updates | Same |
| Cosine minimum fraction | 0.10 of peak | Same |
| AdamW betas / epsilon | (0.9,0.999) / 1e-8 | Same |
| Gradient norm limit | 1.0 | Same |

The cosine schedule spans each arm's prescribed duration. Longer duration thus
changes both update count and decay horizon. The final checkpoint is evaluated;
there is no score-dependent early stopping. Schedule-matched ablations share all
settings other than the pretraining condition listed below.

## Pretraining comparison registry

| Configuration key, for `excerpt` or `full` | SSL | Supervised |
|---|---|---|
| `scratch_<scope>` | None | Clean |
| `maple_<scope>` | Uncorrupted masked SSL | Clean |
| `cora_maple_<scope>` | Corrupted masked SSL | Clean |

Scratch E/F are matched references for the two input scopes. Neither samples
excerpts, because neither performs pretraining. With the same supervised schedules,
input data and paired random seeds their predictions are identical; the two entries
must not be interpreted as independent scientific replications.

Each split needs four full-pool SSL fits: clean/corrupted × excerpt/full. The four
label budgets reuse these same pretrained weights, with a fresh head and separate
supervised optimization for each budget. Initializations, head seeds, source-order,
masking, corruption, and dropout streams are paired across conditions where relevant.
Source splitting precedes any input extraction or fitting. Exact source roles and
label/pool selections are stored in the workspace.

The comparison separates the effect of unlabelled representation learning (MAPLE
versus Scratch) from corruption-aware pretraining (CorA-MAPLE versus MAPLE). It does
not test supervised augmentation. This is a pretraining comparison, not an evaluation of supervised augmentation.

Classical core code remains unmodified. EAC uses attenuation 0.01, normalized
features, event-index time, Euclidean distance; MBW uses the original fixed L2
logistic configuration; WDI retains the original strict frequency >0.5 threshold;
JAC retains its original set comparison; ACM uses native SVC decisions without
probability calibration. Each method uses only its selected labelled records.
