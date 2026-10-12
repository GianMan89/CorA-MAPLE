# Six published perturbation mechanisms and their observation semantics

Six mechanisms from the two cited AFC robustness studies are evaluated. They
represent channel-state failures, event-notification errors, timestamp uncertainty,
and delayed observation. No new perturbation family is claimed.

| Key | Definition | Source | Proposed-method exposure |
|---|---|---|---|
| `false_full` | Select floor(s C) distinct configured channels and set their entire observed state series to 1 | ETFA 2025 full false-alarm setting | SSL only |
| `missing_full` | Select floor(s C) distinct channels and set their entire series to 0 | ETFA 2025 full missing-alarm setting | SSL only |
| `timing` | Select floor(s n) ACT/RTN notifications without replacement, shift times independently by U[-10s,+10s] minutes, clip to original horizon | AFC-RobustBench timing uncertainty | SSL only |
| `late_start` | Remove the first floor(s T) samples, retain the remaining values and rebase their local time | ETFA 2025 late-detection time cropping | Test only |
| `missing` | Delete floor(s n) ACT/RTN notifications uniformly without replacement | AFC-RobustBench missing notifications | Test only |
| `spurious` | Insert floor(s n) notifications with uniformly sampled tag, ACT/RTN type, and timestamp | AFC-RobustBench additional notifications | Test only |

C is the configured channel count (50 for TEP), T the supplied sample count, and n
the number of ACT/RTN transition notifications in the selected input. The budget
is computed for the selected excerpt/episode, not its original parent source.
Timings use the same one-minute grid; timestamp shift parameter is 10 minutes.
The historical full-state snapshot at the first sample is **not a notification**
in the primary protocol, so initially active excerpts do not receive artificial
ACT events added to their event budget. Zero budget leaves the input unchanged.

## Training and testing support

SSL operator choice is uniform among the first three; 20% of views are unchanged.
Corruption-aware pretraining uses severity U[0.05,0.30]. Supervised inputs are always
experimentally uncorrupted. Test severity is 0,0.1,0.2,0.3,0.4,0.5, with three
seeded realizations per positive severity and one shared clean condition.

- **P:** used during CorA-MAPLE pretraining: false_full, missing_full, timing.
- **U:** absent from all training: late_start, missing, spurious.
- 40% and 50% are severity extrapolation only for P operators. On U, the test
  measures cross-operator transfer at the specified severity.

Event loss and channel loss, or message insertion and persistent false activity,
are related physical mechanisms but are not identical interventions. "Test-only"
does not mean unrelated physical concepts never seen in any other form. Equal
numeric severity does not equal equal input damage, information loss or risk.

## Known-snapshot event reconstruction (primary)

The input is a binary historian trace including its initial alarm state. Convert
only its observed state transitions into ACT/RTN notifications. Perturb those
notifications and stable-sort by timestamp and event-order tie break. Initialize
each channel's state from its recorded start snapshot and interpret subsequent
notifications forward:

- An ACT on an inactive tag sets it active; another ACT while active is redundant.
- An RTN on an active tag returns it to normal; an RTN while already inactive is
  redundant and does not invent past activity.
- A missing RTN can legitimately leave a channel active longer. Notification loss
  is not equivalent to forcing a channel inactive.
- A shifted/inserted notification clipped to time zero takes effect at the first
  sample *after* the immutable initial snapshot. It does not overwrite the snapshot.
- An initially active excerpt starts active without inventing a time-zero ACT.

This assumption is part of the observation model: all full TEP records begin
inactive, but many excerpts begin with real active alarms. Blindly allowing leading
RTNs invents history; blindly assuming all excerpts initially inactive also loses
real history. The per-channel snapshot is available in the current binary series
and does not use fault labels or predictions. Its availability for a deployment
must be justified rather than inferred from class knowledge.

The event primitives and classical algorithms are attributed in `vendor/`.
`src/cora_maple/transforms.py` applies the snapshot-aware reconstruction and
valid-length metadata consistently to every evaluated method.

## Delayed observations and padding

The physical retained sequence is X[:,floor(s T):]. Its valid sample length is
T-floor(s T). A numerical tensor may have trailing zeros for batching, but those
positions are **padding**, not known inactive states. Neural attention/pooling/loss
exclude them. Classical baselines are passed the trimmed real sequence. A partial
last q-sample patch includes zero-fill for absent cells and a real-sample weight;
entirely absent patches contribute nothing. Delay therefore changes the information
available, not merely the apparent duration of alarm inactivity.

## Measured input effects

Every test cache records per source, operator, severity and random realization:
valid/lost samples; changed state cells; added/removed active cells; lost/gained
active tags; unchanged-input indicator. Delayed inputs are compared to the retained
clean suffix so that excluded history is not confused with modified activity.
Nominal corruption budgets can have no effect, e.g. selecting an already inactive
tag; that is reported, not discarded. No requirement artificially forces model
scores to decline monotonically with severity.

## References

Manca et al. (2025), *Measuring the Robustness of Alarm Flood Classification Against
Alarm Data Quality Issues*, Proc. IEEE ETFA, 1059–1066.

Manca et al. (2026), *Perturbation-Based Robustness Benchmarking of Online Alarm Flood
Classification under Alarm-Log Degradations and Detection Delays*, SSRN 6999280.

The six-mechanism protocol operationalizes their definitions on the current TEP
binary-state dataset; it is not an exact numerical replication of their different
training budgets, online evaluation horizons, or initial-state assumptions.
