# Native stream-layout contract

PersonaPlex is a duplex delayed-code model, not a single eight-codebook audio
model. The audited native runtime config has 17 global streams:

| Global stream indices | Owner | Native role |
| --- | --- | --- |
| `0` | agent | text |
| `1..8` | agent | Mimi output audio |
| `9..16` | caller | Mimi input audio |

`dep_q=16` is the width of the native depformer audio range (`1..16`); it is
not the number of agent-output audio streams. Consequently, the training
contract requires a serialized `StreamLayout` in every encoded corpus manifest
and rejects any target mask which includes indices `9..16`.

The current adapter stage optimizes only stream `0` and agent audio streams
`1..8`. Caller streams remain condition-only context. Any future model variant
must declare all global streams, prove the groups are disjoint and exhaustive,
and pass the loss-isolation test before training.

This decision was verified against the native runtime routing: caller tokens
are placed in global codebooks `9..16`, agent audio is decoded from `1..8`, and
the model has `K=17` with `dep_q=16`.
