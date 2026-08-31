# HyperLight inside Unicorn's harness

Our conditioning mechanism, rebuilt as a model in
[marmotlab/Unicorn](https://github.com/marmotlab/Unicorn) (MIT), so the claim
that survived this study can be tested in an architecture we did not design.

Only our own code lives here. Unicorn itself is not vendored: `install.py`
regenerates the runner/driver pair from theirs, so re-running it after a `git
pull` in the clone keeps the port in step with their changes.

## Why this exists

Three networks now say the *content* of a structural conditioning code is worth
about nothing (`cityflow4x4_hetero`, `atlanta_1x5`, and the `shrink=0` control on
Ingolstadt21, which ties the unshrunk arm). What keeps winning is having a
hypernetwork at all, and not having a per-intersection index table. That is a
mechanism claim, and the strongest place to test one is someone else's codebase.

HeteroLight is a deliberate choice of host. It **already** feeds every
intersection a fixed-scale structural descriptor — `Tls.int_attr_vec`, 55 dims of
phase type plus per-approach mean length / speed / lane count / link count,
divided by constants that never depend on the loaded roadnet. That is the same
design rule as `transfer/structural.py`. So the information our contract carries
is already in their baseline, as a model *input*, and the open question is not
whether structural information helps but **where it has to enter**.

## The arms

Every arm is HeteroLight with one thing changed: the final head's weights are
generated per intersection instead of shared. `linear_s`, `linear_a`, the
decoder, the GRU and the whole VAE branch are untouched, and `int_vector` still
enters the VAE input in every arm, so nothing is taken away from any of them.

| arm | conditioning code | reads as |
|---|---|---|
| `heterolight` (theirs, unmodified) | — head is shared | structural info as input |
| `structural` | MLP(our 12-feature contract) | the arm under test |
| `learned` | per-intersection embedding | index table; cannot transfer |
| `constant` | MLP(ones) | hypernetwork with no content |

`structural` vs `learned` → does a network-independent code beat an index table.
`structural` vs `constant` → does the content matter.
Any of them vs `heterolight` → does generating the head beat feeding the same
information as input.

## Install

Into a Unicorn checkout (this repo's `external/unicorn_port` → their root):

```bash
cp HyperLight.py       <unicorn>/models/
cp structural_meta.py  <unicorn>/
cp install.py          <unicorn>/
cd <unicorn> && python install.py     # writes runner_hyperlight.py, driver_hyperlight.py
```

Environment for `danielda1/ugat:latest`, which already has SUMO 1.20, torch 2.4
and traci: `pip install einops==0.6.0 ray==2.3.1 pandas scikit-learn 'pydantic<2'`.
The pydantic pin is required — ray 2.3.1 calls `pydantic.fields.ModelField`,
which v2 removed. Set `TRAIN_PARAMS.USE_GPU = False`; their pinned torch
1.13+cu117 does not support this box's RTX 5080 (Blackwell needs 2.7+/CUDA 12.8)
and CPU is fast enough here — ~13 s per 240-decision episode per worker.

## Run

```bash
HYPER_META_MODE=structural  python driver_hyperlight.py   # or learned | constant
HYPER_STRUCT_SHRINK=0.38    ...                           # optional, structural only
python driver_heterolight.py                              # the unmodified control
```

## Caveats to read the numbers with

* **Two of the twelve features are dead here.** Unicorn's
  `ingolstadt_network_21_config.json` carries an empty `neighbor_list`, so
  `neighbor_count` and `controlled_neighbor_ratio` are 0 for every intersection.
  Our own harness sees `neighbor_count[0/0.38/2]` on the same map. Ten features
  vary; two do not.
* **The rest of the contract does port faithfully.** Side by side on
  Ingolstadt21, ours then theirs: `in_lane_count[4/7.43/14]` vs
  `[4/7.52/14]`, `phase_count[2/3.19/4]` vs `[2/3.14/4]`,
  `out_in_lane_ratio[0.5/0.70/1]` vs `[0.5/0.71/1]`. The small gaps are the two
  harnesses enumerating lanes and approaches slightly differently.
* **Numbers from here are on their scale, not ours.** At matched protocol the
  two harnesses differ by ~12 s on MaxPressure (a well-defined algorithm) and
  ~51 s on "fixed time" (which is not one). Compare arms within this harness
  only.
* `shrink` makes the features depend on the loaded network, so a shrunk run is
  an ablation, never a transfer source — same rule as `--structural_shrink`.
