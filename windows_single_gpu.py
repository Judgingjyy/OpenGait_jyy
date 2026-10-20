"""
OpenGait Windows single-GPU launcher.

For one GPU, distributed synchronization is mathematically
unnecessary. Keep the official model and training code unchanged.
"""

import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "opengait"))

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable. Please check PyTorch and NVIDIA driver.")

if torch.cuda.device_count() != 1:
    raise RuntimeError("This launcher is intended for exactly one visible GPU.")

# Single-process distributed compatibility
dist.get_rank = lambda *args, **kwargs: 0
dist.get_world_size = lambda *args, **kwargs: 1
dist.broadcast = lambda tensor, *args, **kwargs: None
dist.barrier = lambda *args, **kwargs: None

def single_all_gather(output, tensor, *args, **kwargs):
    output[0].copy_(tensor)

dist.all_gather = single_all_gather

# Import official entry point
import main as official_main

# Explicitly register DeepGaitV2 on Windows
from modeling.models.deepgaitv2 import DeepGaitV2
official_main.models.DeepGaitV2 = DeepGaitV2

print("[OK] DeepGaitV2 registered")


# Do not wrap single-GPU model in DDP
def identity_ddp(model, *args, **kwargs):
    return model

official_main.get_ddp_module = identity_ddp

import utils
import utils.common
utils.get_ddp_module = identity_ddp
utils.common.get_ddp_module = identity_ddp

import modeling.loss_aggregator as loss_aggregator
loss_aggregator.get_ddp_module = identity_ddp

# Gathering from one GPU is an identity operation
def identity_gather(features, dim=0, requires_grad=True):
    return features

import modeling.base_model as base_model
base_model.ddp_all_gather = identity_gather

import modeling.losses.base as loss_base
loss_base.ddp_all_gather = identity_gather

if __name__ == "__main__":
    options = official_main.opt
    cfg = official_main.config_loader(options.cfgs)

    if options.iter != 0:
        cfg["trainer_cfg"]["restore_hint"] = int(options.iter)
        cfg["evaluator_cfg"]["restore_hint"] = int(options.iter)

    training = options.phase == "train"

    print("========== Windows Single GPU ==========")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Phase:", options.phase)
    print("Config:", options.cfgs)
    print("DDP:", "Disabled (world size = 1)")
    print("========================================")

    official_main.initialization(cfg, training)
    official_main.run_model(cfg, training)
