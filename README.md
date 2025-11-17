# GoeX-Bench 

>[AAAI 2026 Oral] GeoX-Bench: Benchmarking Cross-View Geo-Localization and Pose Estimation Capabilities of Large Multimodal Models

## Highlights

- **GoeX-Bench dataset** – 10,859 panoramic–satellite pairs spanning 128 cities across 49 countries plus 755,976 carefully curated QA pairs (42,900 for benchmarking, the rest for instruction tuning). The dataset merges CVGlobal, CVUSA, OmniCity, LLMGeo, and VIGOR after re-acquiring imagery (2024–2025) to ensure temporal alignment.
- **Seven complementary tasks** – heading estimation (fixed and random locations), map membership tests, intra-map localization, and cross-map retrieval, all illustrated in Fig. 2 of the paper and implemented through the `Tasks/` package.
- **Hydra-driven experiments** – reproducible configuration management in `conf/`, with presets for different model families (Qwen, LLaVA, CloseAI, etc.) and task bundles (location, orientation, map_gauss, etc.).
- **Model abstraction layer** – `model/` contains wrappers for ms-swift engines (`InferenceModel`, `LoRAInferenceModel`) and API clients (`CloseAI`) so that task logic can swap models without rewriting dataloaders.
- **Scalable execution** – `main.py` runs sequential pipelines, `main_pool.py` provides a GPU scheduler for concurrent jobs, and `main_closeai.py` adds deterministic sampling/caching for large CloseAI sweeps.

## Repository Layout

| Path | Description |
| --- | --- |
| `conf/` | Hydra configs for models, tasks, and experiment pipelines (e.g., `conf/config.yaml`, `conf/conf_closeai.yaml`). |
| `Dataset_Index/` | JSON indices and helpers (`data_sampler.py`) for assembling subsets from the five source datasets. |
| `Tasks/` | Modular task implementations built on `task_template.py`, including prompt builders, dataloaders, and writers for each GoeX-Bench task variant. |
| `model/` | Model abstractions for ms-swift engines, LLaVA variants, Qwen interleaved models, and the OpenAI-compatible `CloseAI` client. |
| `main.py` | Default entry point that iterates through Hydra pipelines sequentially. |
| `main_pool.py` | GPU-aware scheduler that runs each `(model, task instance)` in its own process with configurable GPU pools. |
| `main_closeai.py` | CloseAI-centric runner with dataset caching/balancing utilities for multi-task sampling. |
| `reprediction_closeai.py` | Utility to re-run failed/missing samples for CloseAI outputs while reusing cached indices. |
| `outputs/` | Hydra-generated run directories (`YYYY-MM-DD/HH-MM-SS/...`) containing configs, logs, and predictions. |

## Benchmark Tasks

The AAAI PDF formalizes seven evaluation tasks; the code mirrors these through task-specific prompt templates and dataloaders:

1. **Pose Estimation – Fixed Position** (`orin_task`): heading classification assuming the camera is at the satellite map center.
2. **Pose Estimation – Random Position** (`orin_task_random`): heading classification when the camera can be anywhere in the map.
3. **Localization with Prior** (`location_task`): determine whether a ground scene lies within the center crop of a satellite map.
4. **Localization without Prior** (`location_random_task`): same as above but the ground location can appear anywhere within the map.
5. **Intra-Map Localization** (`location_8_gauss`): predict the relative location bucket (e.g., top, bottom, etc.) once the model knows the scene is inside the map.
6. **Cross-Map Retrieval (center prior)** (`map_gauss`): select the correct satellite tile when the match is guaranteed to occur at each map’s center.
7. **Cross-Map Retrieval (random)** (`map_gauss_random`): select the correct tile when the camera can be off-center in any candidate map.

Each task returns structured JSON (`{"explanation": ..., "direction": ...}`, `{"is_in_range": ...}`, etc.) so that downstream metrics align with the paper’s evaluation protocol.

## Data Preparation

1. **Download / license source datasets** – CVGlobal, CVUSA, OmniCity, LLMGeo, and VIGOR. Follow the acquisition guidelines in the PDF (north-aligned panoramas, 512×512 aerial mosaics, synchronized capture dates).
2. **Create local mirrors** – Place imagery under a common root such as `/mnt/dataset/...`. Update the `root_dir` fields in your Hydra task configs accordingly.
3. **Use Dataset_Index JSONs** – `Dataset_Index/100Sample/*.json` demonstrate the schema expected by `Tasks.*Dataset` classes. Customize or regenerate them via `Dataset_Index/data_sampler.py` if you need different sample counts.
4. **Caching balanced splits** – `main_closeai.py` builds `dataset_indices_cache.json` inside `output_base/` to make sure each task draws balanced categories (orientation bins, map IDs, etc.). Reuse or point `previous_cache_path` to avoid duplicates between runs.

## Environment Setup

```bash
conda create -n geoxbench python=3.10 -y
conda activate geoxbench

# Core dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install hydra-core==1.3.2 omegaconf==2.3.0 tqdm pillow aiohttp openai ms-swift accelerate einops transformers datasets

# Task-specific extras (install as needed)
pip install llava swift vision pillow scikit-image shapely
```

- `model/CloseAI.py` now expects sensitive credentials via environment variables. Set `export CLOSEAI_BASE_URL=...` and `export CLOSEAI_API_KEY=...` (or specify these fields in the Hydra model config) before invoking any CloseAI pipelines.
- Ensure the Google Maps API tooling mentioned in the manuscript is available if you plan to re-acquire imagery; otherwise, reuse the cached tiles committed to your data mirrors.

## Configuration Model

Hydra composes experiments from three blocks (see `conf/config.yaml` and `conf/conf_closeai.yaml`):

```yaml
defaults:
  - model@models.0: qwen25vl_7b
  - model@models.1: mPLUG-owl3_7b
  - task@tasks.location: location
  - task@tasks.orin: orin
  - task@tasks.orin_random: orin_random

output_base: /home/GeoX-Bench/output
pipelines:
  - model: ${models.0}
    tasks:
      - ${tasks.location}
      - ${tasks.orin}
```

- **Models** describe the inference backend (`class_path`, checkpoints, batch size, GPU occupancy, optional LoRA checkpoints, etc.).
- **Tasks** bind dataset classes (`Tasks.location_task.InRangeDataset`), data writers, message templates, and dataset instances (JSON path, sample size, output name).
- **Pipelines** pair a model with one or more tasks; each task can expose multiple `instances` to target different datasets or splits.

Override anything at runtime with standard Hydra syntax, e.g., `python main.py output_base=/mnt/exp/outputs model@models.0=qwen25vl_3b`.

## Running Experiments

### Sequential pipeline (default)

```bash
python main.py \
  --config-name conf_random8_AAAI___ \
  output_base=/mnt/experiments/CloseAI
```

`main.py` spawns one subprocess per pipeline, iterates through each task instance, and writes JSON predictions to `${output_base}/${task.output_dir}/${model_name}-${instance.output}`.

### GPU-pooled execution

```bash
python main_pool.py \
  --config-name conf_mapgauss \
  scheduler.gpu_pool=[0,1,2,3]
```

This entry point allocates a dedicated process (and GPU slice) to each `(model, task instance)` so long-running jobs do not block others. Use `pipe.model.gpu_occu` in your configs to state how many GPUs a model consumes.

### CloseAI balanced sweeps

```bash
export CLOSEAI_BASE_URL=https://api.example.com/v1
export CLOSEAI_API_KEY=sk-...

python main_closeai.py \
  --config-name conf_closeai \
  target_samples=256 \
  force_regenerate=false
```

`main_closeai.py` caches balanced indices per task/dataset ID, detects previously sampled entries, and reuses caches stored in `${output_base}/dataset_indices_cache.json`. Use `previous_cache_path` to enforce disjoint samples across launches.

### Re-running missing items

If a CloseAI job partially failed, use `reprediction_closeai.py input_glob='outputs/2025-05-*/**/*.json'` to backfill missing predictions without reprocessing entire datasets.

## Outputs & Evaluation

- Hydra writes every run to `outputs/<YYYY-MM-DD>/<HH-MM-SS>/` along with `.hydra/{config,hydra,overrides}.yaml` snapshots.
- Per-task writers embed model predictions under `prediction.message` and parse task-specific signals (direction, is_in_range, etc.) into `prediction.prediction_element` for easy metric computation.
- For AAAI tables, aggregate predictions in the same order as described in the PDF: compute accuracy for heading estimation, true-positive rates for map membership, and top-1 retrieval accuracy for cross-map selection. Use the provided JSON schema to plug into your evaluation notebooks.

## Reproducing the AAAI Experiments

1. **Align datasets** – follow the pre-processing pipeline in Fig. 3 of the PDF (north alignment, FoV slicing, synchronized satellite refresh).
2. **Select configs** – `conf/conf_random8_AAAI___.yaml`, `conf/conf_random8_AAAI.yaml`, and `conf/conf_closeai*.yaml` replicate the sweeps reported in Table 1 of the manuscript.
3. **Run models** – execute `main.py` (for ms-swift backbones) or `main_closeai.py` (for API-based models) per section above. Ensure batch sizes and GPU occupancy match the paper.
4. **Collect metrics** – parse `outputs/**/model-task.json` files to build per-task accuracy tables, mirroring the columns IM Rnd., CM Std., CM Rnd., etc.
5. **Document settings** – the AAAI template requires reporting dataset versions, prompt templates, and inference budgets. 

## Troubleshooting & Tips

- **Credential management** – `CloseAI` no longer ships with hard-coded API secrets. Define `CLOSEAI_BASE_URL` and `CLOSEAI_API_KEY` (or edit your Hydra config) before launching `main_closeai.py`.
- **File handles** – the dataloaders copy each sample into a temporary directory (`mkdtemp`). Failing to consume `batch["temp_dirs"]` in writers leads to leaked disk usage. Always let `DataWriter` clean up temp folders, or call `cleanup_temp_dirs` manually if you build custom writers.
- **GPU visibility** – when using `main_pool.py`, ensure `torch.cuda.device_count()` reflects the hardware you intend to use. Debug by exporting `GPU_POOL` or setting `scheduler.gpu_pool` explicitly.
- **Hydra overrides** – pass `+task@tasks.new_task=conf/task/...yaml` to experiment with additional task definitions without touching committed configs.


For questions or collaboration inquiries, please open an issue or contact the maintainers listed in the manuscript once the review phase ends.
