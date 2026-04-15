# Experiments

This file records model lines that were tested and whether they are still worth carrying in the codebase.

## Active

### `ridge`
- type: polynomial ridge over engineered eye/head features
- status: keep
- reason: cheap benchmark, interpretable, still competitive on easy splits

### `frame_wide_aug_long`
- type: wide eye-crop CNN + head/face features + concat fusion
- status: keep
- reason: best known region generalization
- best known strict result:
  - `region_holdout`: `97.5px`

### `frame_attention_matched_long`
- type: parameter-matched eye-crop CNN + head/face features + token attention fusion
- status: keep
- reason: best known session generalization
- best known strict result:
  - `session_holdout`: `191.3px`

## Discarded

### `frame_base`
- type: smaller concat eye-crop model
- status: discard
- reason: dominated by the wider concat model

### `frame_base_aug`
- type: smaller concat eye-crop model with augmentation
- status: discard
- reason: dominated by the wider concat model

### `frame_wide_aug`
- type: intermediate training schedule for the wide concat model
- status: discard
- reason: replaced by `frame_wide_aug_long`

### `frame_attention_long`
- type: oversized attention fusion model
- status: discard
- reason: comparison was parameter-confounded; replaced by the matched attention model

### `hybrid_ridge_x_frame_wide_y`
- type: ridge for `x`, frame model for `y`
- status: discard
- reason: did not justify its extra complexity

### `temporal_matched_long`
- type: parameter-matched GRU over capture sequences
- status: discard
- reason: even after parameter matching, it was not competitive enough to justify its training cost
- parameter count: `201,138`
- result:
  - `label_holdout`: `225.6px`

## Notes

- Earlier full strict comparisons established the current frame tradeoff:
  - `session_holdout`: `frame_attention_matched_long` `191.3px`, `frame_wide_aug_long` `215.1px`, `ridge` `271.4px`
  - `region_holdout`: `frame_wide_aug_long` `97.5px`, `frame_attention_matched_long` `123.6px`, `ridge` `148.0px`
- Latest active-model `label_holdout` comparison:
  - `ridge` `90.1px`
  - `frame_wide_aug_long` `92.4px`
  - `frame_attention_matched_long` `100.1px`
- Latest retrain + quick cross-session check on the larger `15`-session dataset (`6407` train / `1598` eval):
  - refreshed `concat` checkpoint trainer eval: `82.6px x / 71.2px y`
  - `session_holdout` `quick`: `frame_attention_matched_long` `220.0px`, `frame_wide_aug_long` `224.2px`, `ridge` `239.1px`
  - implication: `concat` remains the cleaner main line for easy/region behavior, but `attention` is still live on cross-session robustness and has not been ruled out
