#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gpu_scheduler.py – schedule every (model, inst) on its own set of GPUs.

New in this version
-------------------
* A *GPU-pool* mechanism lets you restrict the scheduler to an arbitrary
  subset of physical devices via Hydra (`scheduler.gpu_pool=[…]`), a YAML
  config entry, or the environment variable `GPU_POOL`.
"""
from __future__ import annotations
import os
import time
import signal
import torch
from multiprocessing import Process
from pathlib import Path
from importlib import import_module

import hydra
from omegaconf import DictConfig, OmegaConf

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _resolve(dotted: str):
    mod, attr = dotted.rsplit(".", 1)
    return getattr(import_module(mod), attr)


def _instance_worker(model_cfg, task_cfg, inst_cfg, global_cfg,
                     batch_size, gpu_ids):
    """Runs inside a child process and handles exactly ONE (model, inst)."""
    # 1. fence the GPUs for this child
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    # 2. build the model
    Model = _resolve(model_cfg.class_path)
    model = Model(model_cfg)

    # 3. delegate to the unchanged helper from the main script
    _run_instance = _resolve("main._run_instance")
    _run_instance(model, inst_cfg, task_cfg, global_cfg, batch_size)

    # 4. goodbye – process exit will reclaim memory, but be polite
    del model
    torch.cuda.empty_cache()


def _get_gpu_pool(cfg: DictConfig) -> list[int]:
    """
    Determine which physical GPU IDs the scheduler is allowed to use.

    Priority (first one that is set wins):
        1. cfg.scheduler.gpu_pool          (Hydra CLI override or YAML)
        2. $GPU_POOL environment variable  (comma-separated list)
        3. all devices                     (default)
    """
    # 1) explicit Hydra/YAML entry
    if "scheduler" in cfg and cfg.scheduler is not None:
        pool = cfg.scheduler.get("gpu_pool")
        if pool:
            return list(map(int, pool))

    # 2) environment variable
    env = os.getenv("GPU_POOL")
    if env:
        return [int(x) for x in env.split(",") if x.strip()]

    # 3) default = every card the driver sees
    return list(range(torch.cuda.device_count()))


# --------------------------------------------------------------------------- #
# Scheduler                                                                   #
# --------------------------------------------------------------------------- #
@hydra.main(config_path="conf", config_name="conf_mapgauss", version_base="1.3")
def main(cfg: DictConfig):

    # -------- figure out what GPUs we may touch -----------------------------
    physical_total = torch.cuda.device_count()
    gpu_pool       = _get_gpu_pool(cfg)

    # sanity checks
    bad = [i for i in gpu_pool if i < 0 or i >= physical_total]
    if bad:
        raise RuntimeError(
            f"Invalid GPU id(s) {bad}. This machine only has "
            f"{physical_total} device(s): {list(range(physical_total))}"
        )
    if len(set(gpu_pool)) != len(gpu_pool):
        raise RuntimeError("Duplicate entries found in gpu_pool!")

    free_gpus = sorted(gpu_pool)          # stack of idle ids
    running   = []                        # [(Process, [ids]), …]

    Path(cfg.output_base).mkdir(parents=True, exist_ok=True)
    config_path = Path(cfg.output_base) / "config.yaml"

    with config_path.open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    if len(gpu_pool) == physical_total:
        print(f"Detected {physical_total} GPUs → using all ({free_gpus})")
    else:
        print(
            f"Detected {physical_total} GPUs but scheduler restricted to "
            f"pool {gpu_pool}"
        )

    # ------------------- build the job queue --------------------------------
    jobs = []      # [(model_cfg, task_cfg, inst_cfg, batch_size, occu), …]

    for pipe in cfg.pipelines:
        occu = getattr(pipe.model, "gpu_occu", 1)
        pipe.model.batch_size = 1
        for task_cfg in pipe.tasks:
            for inst_cfg in task_cfg.instances:
                jobs.append((
                    pipe.model,
                    task_cfg,
                    inst_cfg,
                    pipe.model.batch_size,
                    occu,
                ))

    # ------------------- schedule them --------------------------------------
    for (model_cfg, task_cfg, inst_cfg, bs, occu) in jobs:

        if occu > len(gpu_pool):
            raise RuntimeError(
                f"Job {model_cfg.name}/{inst_cfg.output} needs {occu} GPUs "
                f"but only {len(gpu_pool)} are available in the pool {gpu_pool}."
            )

        # wait until enough GPUs are free -------------------
        while True:
            for proc, ids in running[:]:
                if not proc.is_alive():                 # finished → reclaim
                    proc.join()
                    running.remove((proc, ids))
                    free_gpus.extend(ids)
                    free_gpus.sort()
            if len(free_gpus) >= occu:
                break
            time.sleep(1)

        # allocate & launch -------------------------------
        alloc = [free_gpus.pop(0) for _ in range(occu)]
        tag   = f"{model_cfg.name}:{inst_cfg.output}"
        print(f"Launching {tag} on GPUs {alloc}")

        p = Process(
            target=_instance_worker,
            args=(model_cfg, task_cfg, inst_cfg, cfg, bs, alloc),
            daemon=False,
        )
        p.start()
        running.append((p, alloc))

    # ------------------- wait for everything to finish ----------------------
    for proc, _ in running:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError("One of the jobs failed (see above).")

    print("✅  All instances completed.")


if __name__ == "__main__":
    # Make Ctrl-C propagate to children
    signal.signal(signal.SIGINT,  signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    main()