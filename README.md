# Type Cast

Offline generation driver for the *Type Cast (I–VIII)* installation:
synchronized 5″ video walls showing classic-Hollywood archetypes being
deconstructed by progressive attention bending of an underlying
generative video model.

This repo holds **two** independent components:

* **`webui/`** + **`kiosk.sh`** + **`SETUP.md`** — the Pi 3B+ kiosk
  side. Boots into Chromium fullscreen, loops the per-screen video
  produced by the generator. No Python.
* **`generator/`** — the offline experiment runner. Loads an experiment
  YAML, walks a prompt × deformation-band matrix, drives the
  `scope-attention-bender` LongLive pipeline, encodes each result to
  Pi-optimal MP4. Python.

## Generator quick start

### Install

```bash
# in the env that has scope + scope-attention-bender installed
pip install -e .                                          # this repo
pip install -e /path/to/scope-attention-bender            # bender hooks
```

`scope-attention-bender` isn't a hard dependency because it pulls
`scope` + `torch` + CUDA — a heavy chain only the generation box needs.
Install it manually in the env where you actually generate.

### Validate an experiment YAML (no GPU)

```bash
type-cast validate experiments/cowboy_femme_ffn_output_scale_0.yaml
```

Prints the resolved spec + the matrix size estimate. Fast feedback when
editing experiments.

### Dry-run (no GPU, no model load, no I/O)

```bash
type-cast run experiments/cowboy_femme_ffn_output_scale_0.yaml --dry-run
```

Walks the full matrix with a stub adapter, logging every operation that
would happen. Use to validate naming + folder layout + sweep math before
burning GPU minutes.

### Real run

```bash
# Default GPU
type-cast run experiments/cowboy_femme_ffn_output_scale_0.yaml

# Specific GPU (when Scope is running on cuda:0, target cuda:1)
type-cast run experiments/cowboy_femme_ffn_output_scale_0.yaml --device cuda:1

# Custom output root
type-cast run experiments/x.yaml --output /mnt/external/type_cast_runs/
```

Output lands under `output/<experiment_name>_<timestamp>/`:

```
output/type_cast_v1_ffn_output_scale_0_20260613-190134/
├── manifest.json                         # full kwargs + bender SHA + timestamps
├── prompt-01_cowboy/
│   ├── prompt-01_band-00.mp4             # FFN layers 0,1,2 bent
│   ├── prompt-01_band-01.mp4             # FFN layers 1,2,3 bent
│   ├── ...
│   ├── prompt-01_band-27.mp4             # FFN layers 27,28,29 bent
│   ├── prompt-01_baseline.mp4            # no bend (sweep position: end)
│   └── prompt-01_merged.mp4              # concat-copy of all the above
└── prompt-02_femme/
    └── ...
```

## Experiment YAML schema

See `experiments/cowboy_femme_ffn_output_scale_0.yaml` for a fully-
commented example.

* `name`, `model`, `pipeline_init`, `video`, `prompts`, `bending_base`,
  `sweep`, `baseline`, `merge`
* `bending_base` is the **deformation** — applied to every video in the
  experiment. e.g. `{ bending_enabled: true, ffn_output_scale: 0.0 }`
  means "scale the FFN output to 0".
* `sweep.kind: ffn_layer` slides a `window`-wide band of FFN layers
  across the model with stride `stride`. V1 implements this kind only;
  the protocol in `generator/sweep.py` is extensible.
* `baseline.per_prompt: true` + `position: end` writes one no-bend
  video per prompt at the end of the sweep — loops naturally back into
  band-00 on the kiosk.
* `merge: true` concat-copies (no re-encode) all per-prompt videos into
  one merged mp4. Useful for a single-stream kiosk player.

## Tests

```bash
python -m pytest tests/ -q
```

72 tests cover sweep math, YAML parsing, runner matrix iteration,
encode cmdlines (mocked subprocess), naming, manifest.

## Multi-GPU concurrency

Each LongLive pipeline instance loads ~20 GB. On a multi-GPU box you
can run the bender's WebUI workflow (Infinite Jest, etc.) on `cuda:0`
in parallel with Type Cast on `cuda:1` — independent Python processes,
no shared state. Single-GPU boxes need Scope's pipeline unloaded
before generating.

## Bender integration

Type Cast hooks into `scope-attention-bender` through exactly two seams:

1. `scope_attention_bender.pipelines.longlive_pipeline.AttentionBenderLongLivePipeline`
   constructed with `start_webui=False` and the AR knobs
   (`num_frame_per_block`, `local_attn_size`, `sink_size`).
2. `scope_attention_bender.orchestrator.autoregressive_driver.generate_autoregressive_video`
   for the chunk-loop-to-target-frames helper.

Both are additive bender contributions — the LTX2.3 pipeline path
(used by the Infinite Jest installation) is untouched.
