
import os
import sys
import json
from pathlib import Path

import numpy as np
import yaml
import torch.distributed as dist

# 1. 项目与配置路径
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

sys.path.insert(0, str(ROOT / "opengait"))

CFG_PATH = (
    ROOT / "configs" / "lidargait"
    / "lidargait_sustech1k_depthonly.yaml"
)

print("[1] 加载 YAML 配置", flush=True)

with open(CFG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

data_cfg = cfg["data_cfg"]

dataset_root = Path(data_cfg["dataset_root"])
partition_path = Path(data_cfg["dataset_partition"])

assert dataset_root.is_dir(), "数据集目录不存在"
assert partition_path.is_file(), "官方划分文件不存在"
assert data_cfg["data_in_use"] == [0], "模态索引错误"

print("数据集:", dataset_root)
print("模态索引:", data_cfg["data_in_use"])

# 2. 检查身份划分
print("\n[2] 检查官方身份划分", flush=True)

with open(partition_path, "r", encoding="utf-8") as f:
    partition = json.load(f)

actual_ids = {
    p.name for p in dataset_root.iterdir()
    if p.is_dir()
}

train_ids = set(partition["TRAIN_SET"])
test_ids = set(partition["TEST_SET"])

missing_train = train_ids - actual_ids
missing_test = test_ids - actual_ids

print("训练身份:", len(train_ids & actual_ids), "/", len(train_ids))
print("测试身份:", len(test_ids & actual_ids), "/", len(test_ids))
print("缺失训练身份:", len(missing_train))
print("缺失测试身份:", len(missing_test))

if missing_train or missing_test:
    raise RuntimeError(
        "数据集身份不完整，请先检查提取和解压结果。"
    )

# 3. 初始化单进程分布式环境
# OpenGait 的 get_msg_mgr() 需要已初始化的进程组
print("\n[3] 初始化数据加载环境", flush=True)

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29731"

dist.init_process_group(
    backend="gloo",
    init_method="env://",
    rank=0,
    world_size=1
)

try:
    from utils import get_msg_mgr
    from data.dataset import DataSet
    from data.transform import get_transform

    get_msg_mgr().init_logger(
        str(ROOT / "output"),
        log_to_file=False
    )

    # 4. 使用官方 DataSet
    print("\n[4] 构建官方 DataSet", flush=True)

    train_dataset = DataSet(data_cfg, training=True)

    print("训练序列数:", len(train_dataset))
    print("训练身份数:", len(train_dataset.label_set))

    assert len(train_dataset) > 0, "训练集为空"

    # 5. 读取一条真实数据
    print("\n[5] 读取真实步态序列", flush=True)

    data_list, seq_info = train_dataset[0]

    print("序列信息:", seq_info[:3])
    print("模态数量:", len(data_list))

    assert len(data_list) == 1, "模态数量不是1"

    depth = np.asarray(data_list[0])

    print("原始数据Shape:", depth.shape)
    print("原始数据dtype:", depth.dtype)
    print("最小值:", depth.min())
    print("最大值:", depth.max())

    # 6. 调用官方预处理
    print("\n[6] 测试官方 Transform", flush=True)

    transform_cfg = cfg["trainer_cfg"]["transform"][0]
    transform = get_transform(transform_cfg)

    depth_processed = transform(depth)

    print("预处理后Shape:", depth_processed.shape)
    print("最小值:", depth_processed.min())
    print("最大值:", depth_processed.max())

    if depth_processed.ndim != 4:
        raise ValueError(
            "预处理输出不是4维，需要检查PKL格式"
        )

    if depth_processed.shape[1] != 3:
        raise ValueError(
            "通道数不是3，与官方ResNet9配置不一致"
        )

    print("\n[OK] 官方 DataSet 读取成功")
    print("[OK] 官方 Transform 执行成功")
    print("[OK] 输入维度符合当前配置")

finally:
    dist.destroy_process_group()
