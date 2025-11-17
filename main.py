#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from importlib import import_module
from pathlib import Path
from typing import Any
import torch
import pdb
import gc
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Process


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _resolve(dotted: str) -> Any:
    mod, attr = dotted.rsplit(".", 1)
    return getattr(import_module(mod), attr)


def _run_instance(model, inst_cfg, task_cfg, global_cfg, batch_size):
    DataWriter      = _resolve(task_cfg.data_writer_class)
    LMMMessage      = _resolve(task_cfg.lmm_message_class)
    Dataset         = _resolve(task_cfg.dataset_class)
    get_dataloader  = _resolve(task_cfg.get_dataloader_fn)

    task_folder = os.path.join(global_cfg.output_base, task_cfg.output_dir)
    os.makedirs(task_folder, exist_ok=True)
    writer_output_path = os.path.join(task_folder,
                                      model.config.name + '-' + inst_cfg.output)

    writer = DataWriter(writer_output_path)

    dl = get_dataloader(
        root_dir       = inst_cfg.root_dir,
        json_path      = inst_cfg.json_path,
        batch_size     = batch_size,
        sample_size    = inst_cfg.sample_size,
        message_formatter = LMMMessage(),
        dataset        = Dataset,
    )

    for batch in tqdm(dl, desc=f"{task_cfg.name}:{inst_cfg.output}"):
        writer(model(batch["message"]), batch["instance"], batch["temp_dirs"])
    writer.write()

    # Explicitly clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()  # Force garbage collection



def run_pipeline(pipe, cfg):
    # exactly your existing per-pipeline logic:
    Model = _resolve(pipe.model.class_path)
    model = Model(pipe.model)
    for t_cfg in pipe.tasks:
        for inst in t_cfg.instances:
            _run_instance(model, inst, t_cfg, cfg, pipe.model.batch_size)
    # cleanup inside child (mostly redundant; process exit will do it)
    del model
    torch.cuda.empty_cache()

@hydra.main(config_path="conf", config_name="conf_random8_AAAI___", version_base="1.3")
def main(cfg: DictConfig) -> None:
    Path(cfg.output_base).mkdir(parents=True, exist_ok=True)

    for pipe in cfg.pipelines:
        print(f"Spawning process for pipeline: {pipe.model.name}")
        p = Process(target=run_pipeline, args=(pipe, cfg))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Pipeline {pipe.model.name} failed with exit code {p.exitcode}")

if __name__ == "__main__":
    main()
