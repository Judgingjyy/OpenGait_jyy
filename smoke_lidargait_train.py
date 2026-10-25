
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

sys.path.insert(0, str(ROOT / "opengait"))

# 关闭 libuv，解决之前的 Windows 报错
os.environ["USE_LIBUV"] = "0"

CFG_PATH = (
    ROOT / "configs" / "lidargait"
    / "lidargait_sustech1k_depthonly.yaml"
)

def main():
    if not CFG_PATH.is_file():
        raise FileNotFoundError(CFG_PATH)

    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到 CUDA GPU")

    print("[1] GPU:", torch.cuda.get_device_name(0), flush=True)

    # Windows 单进程通信环境
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29732?use_libuv=0",
        rank=0,
        world_size=1
    )

    mgr = None

    try:
        from utils import (
            config_loader,
            get_msg_mgr,
            init_seeds
        )
        from modeling import models

        print("[2] 加载训练配置", flush=True)

        cfg = config_loader(str(CFG_PATH))
        trainer = cfg["trainer_cfg"]

        assert cfg["data_cfg"]["data_in_use"] == [0]
        assert trainer["sync_BN"] is False
        assert trainer["with_test"] is False

        init_seeds(42)

        output_dir = (
            ROOT / "output"
            / cfg["data_cfg"]["dataset_name"]
            / cfg["model_cfg"]["model"]
            / trainer["save_name"]
        )

        mgr = get_msg_mgr()
        mgr.init_manager(
            str(output_dir),
            True,
            trainer["log_iter"],
            0
        )

        print("[3] 初始化 LiDARGait 模型", flush=True)

        model_name = cfg["model_cfg"]["model"]
        Model = getattr(models, model_name)

        # 使用官方网络，不包装 DDP
        model = Model(cfg, training=True)

        n_params = sum(
            p.numel() for p in model.parameters()
        )

        print("模型:", model_name)
        print("参数量(M):", round(n_params / 1e6, 3))
        print("训练步数:", trainer["total_iter"])
        print("Batch:", trainer["sampler"]["batch_size"])

        print("[4] 开始冒烟训练", flush=True)

        # 调用 OpenGait 原有训练循环
        Model.run_train(model)

        print("[5] 训练循环结束", flush=True)
        print("完成 Iterations:", model.iteration)

        ckpt = (
            output_dir
            / "checkpoints"
            / f"{trainer['save_name']}-"
              f"{model.iteration:05d}.pt"
        )

        if model.iteration < trainer["total_iter"]:
            raise RuntimeError("训练未完成预定步数")

        if not ckpt.is_file():
            raise RuntimeError(f"没有找到权重文件: {ckpt}")

        print("[OK] Checkpoint:", ckpt)
        print("[OK] LiDARGait 冒烟测试完成")

    finally:
        if mgr is not None and hasattr(mgr, "writer"):
            mgr.writer.close()

        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
