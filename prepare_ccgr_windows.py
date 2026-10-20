from pathlib import Path
import json
import pickle
import numpy as np
import yaml

# ============================
# 1. 查找 CCGR-MINI 数据根目录
# ============================

base = Path(r"R:\Datasets\CCGR-MINI")
split_path = Path("datasets/CCGR-MINI/CCGR-MINI.json")
source = Path("configs/deepgaitv2/DeepGaitV2_sustech1k.yaml")

for path in (base, split_path, source):
    if not path.exists():
        raise SystemExit(f"必要文件不存在: {path}")

partition = json.loads(split_path.read_text(encoding="utf-8"))

candidates = [base]

for p in base.iterdir():
    if p.is_dir() and not p.name.isdigit():
        candidates.append(p)
        for sub in p.iterdir():
            if sub.is_dir() and not sub.name.isdigit():
                candidates.append(sub)

root = None

for candidate in candidates:
    hits = sum(
        (candidate / pid).is_dir()
        for pid in partition["TRAIN_SET"][:30]
    )
    if hits >= 10:
        root = candidate
        break

if root is None:
    print("数据目录前几项:")
    for p in list(base.iterdir())[:15]:
        print(" ", p.name)
    raise SystemExit(
        "没有找到 OpenGait 要求的 ID/序列/视角/PKL 结构。"
    )

# ============================
# 2. 找到 Silhouette 文件下标
# ============================

sample_file = None
sil_index = None

for pid in partition["TRAIN_SET"]:
    person = root / pid
    if not person.is_dir():
        continue

    for seq in person.iterdir():
        if not seq.is_dir():
            continue

        for view in seq.iterdir():
            if not view.is_dir():
                continue

            files = sorted(p for p in view.iterdir() if p.is_file())

            for i, file in enumerate(files):
                if file.suffix.lower() == ".pkl" and "sil" in file.name.lower():
                    sample_file = file
                    sil_index = i
                    break

            if sample_file:
                break
        if sample_file:
            break
    if sample_file:
        break

if sample_file is None:
    raise SystemExit("没有找到 Silhouette PKL，请检查预处理格式。")

with open(sample_file, "rb") as f:
    sample = np.asarray(pickle.load(f))

print("Silhouette 示例:", sample_file)
print("Shape:", sample.shape)
print("Dtype:", sample.dtype)

if sample.ndim != 3 or sample.shape[-1] not in (64, 128):
    raise SystemExit(
        "Silhouette 形状不符合本配置预期，请检查尺寸。"
    )

# 再检查一些序列的模态排序
checked = 0

for pid in partition["TRAIN_SET"][:15]:
    person = root / pid
    if not person.is_dir():
        continue

    for seq in person.iterdir():
        if not seq.is_dir():
            continue

        for view in seq.iterdir():
            if not view.is_dir():
                continue

            files = sorted(p for p in view.iterdir() if p.is_file())

            if not files:
                continue

            if sil_index >= len(files) or "sil" not in files[sil_index].name.lower():
                raise SystemExit(
                    f"模态顺序不一致，问题目录: {view}"
                )

            checked += 1
            if checked >= 12:
                break
        if checked >= 12:
            break
    if checked >= 12:
        break

train_ids = sum(
    (root / pid).is_dir()
    for pid in partition["TRAIN_SET"]
)
test_ids = sum(
    (root / pid).is_dir()
    for pid in partition["TEST_SET"]
)

print("训练身份:", train_ids)
print("测试身份:", test_ids)

if train_ids < 500 or test_ids < 300:
    raise SystemExit("身份数量不足，请检查数据是否完整。")

# ============================
# 3. 生成 YAML
# ============================

cfg = yaml.safe_load(source.read_text(encoding="utf-8"))

cfg["data_cfg"].update({
    "dataset_name": "CCGR-MINI",
    "test_dataset_name": "CCGR-MINI",
    "dataset_root": str(root),
    "dataset_partition": str(split_path).replace("\\", "/"),
    "data_in_use": [sil_index],
    "num_workers": 0,
    "cache": False,
})

cfg["model_cfg"]["SeparateBNNecks"]["class_num"] = train_ids

cfg["trainer_cfg"].update({
    "total_iter": 2000,
    "save_iter": 500,
    "log_iter": 50,
    "restore_hint": 0,
    "save_name": "DGv2_CCGR_Win_2k",
    "sync_BN": False,
    "with_test": False,
    "enable_float16": True,
})

cfg["trainer_cfg"]["sampler"].update({
    "batch_size": [2, 2],
    "frames_num_fixed": 8,
    "frames_skip_num": 2,
})

cfg["optimizer_cfg"]["lr"] = 0.03
cfg["scheduler_cfg"]["milestones"] = [1200, 1700]

cfg["evaluator_cfg"].update({
    "eval_func": "evaluate_CCGR_MINI",
    "restore_hint": 2000,
    "save_name": "DGv2_CCGR_Win_2k",
    "enable_float16": True,
})

cfg["evaluator_cfg"]["sampler"].update({
    "batch_size": 1,
    "frames_all_limit": 48,
})

out = Path("configs/deepgaitv2/DeepGaitV2_CCGR_Windows.yaml")

out.write_text(
    yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
    encoding="utf-8"
)

print("\n========== YAML 生成成功 ==========")
print("数据:", root)
print("模态下标:", sil_index)
print("模型:", cfg["model_cfg"]["model"])
print("Batch:", cfg["trainer_cfg"]["sampler"]["batch_size"])
print("训练步数:", cfg["trainer_cfg"]["total_iter"])
print("YAML:", out)
print("===================================")
