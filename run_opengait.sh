
#!/usr/bin/env bash
set -Eeuo pipefail

# 用法:
# bash run_opengait.sh <配置文件> [train|test] [checkpoint迭代数]

cd "$(dirname "$0")"

CFG="${1:-}"
PHASE="${2:-train}"
ITER="${3:-}"
NPROC="${NPROC_PER_NODE:-1}"

if [[ -z "$CFG" || ! -f "$CFG" ]]; then
    echo "错误：请提供存在的 YAML 配置文件"
    exit 1
fi

if [[ "$PHASE" != "train" && "$PHASE" != "test" ]]; then
    echo "运行模式必须是 train 或 test"
    exit 1
fi

if [[ -n "$ITER" && ! "$ITER" =~ ^[0-9]+$ ]]; then
    echo "Checkpoint 迭代数必须为非负整数"
    exit 1
fi

if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE 必须是正整数"
    exit 1
fi

export PYTHONPATH="$PWD/opengait${PYTHONPATH:+:$PYTHONPATH}"

# 1. 通过实际配置自动计算输出路径
OUT=$(python - "$CFG" "$PHASE" <<'PY'
import sys
from pathlib import Path
from utils import config_loader

cfg = config_loader(sys.argv[1])
phase = sys.argv[2]

data = cfg["data_cfg"]
if isinstance(data, list):
    dataset = "_".join(d["dataset_name"] for d in data)
else:
    dataset = data["dataset_name"]

model = cfg["model_cfg"]["model"]
engine = cfg["trainer_cfg"] if phase == "train" else cfg["evaluator_cfg"]
save_name = engine["save_name"]

print(Path("output") / dataset / model / save_name)
PY
)

# 2. 每次运行创建独立记录文件夹
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
RECORD="$OUT/records/${PHASE}_${RUN_ID}"

mkdir -p "$RECORD"

echo "实验目录: $OUT"
echo "记录目录: $RECORD"

# 3. 保存原始配置和默认配置
cp "$CFG" "$RECORD/config_used.yaml"
cp "configs/default.yaml" "$RECORD/default.yaml"

# 保存合并后的实际生效配置
python - "$CFG" "$PHASE" "$ITER" "$RECORD" <<'PY'
import sys
from pathlib import Path
import yaml
from utils import config_loader

config_path, phase, iteration, record = sys.argv[1:]
cfg = config_loader(config_path)

# 与 OpenGait main.py 的 --iter 覆盖逻辑保持一致
if iteration and int(iteration) != 0:
    cfg["trainer_cfg"]["restore_hint"] = int(iteration)
    cfg["evaluator_cfg"]["restore_hint"] = int(iteration)

path = Path(record) / "effective_config.yaml"

with open(path, "w", encoding="utf-8") as f:
    yaml.safe_dump(
        cfg, f,
        allow_unicode=True,
        sort_keys=False
    )
PY

# 4. 保存环境与代码信息
python -m pip freeze > "$RECORD/pip_freeze.txt"

nvidia-smi > "$RECORD/gpu_info.txt" 2>&1 || true

git rev-parse HEAD \
    > "$RECORD/git_commit.txt" 2>/dev/null || true

git status --short \
    > "$RECORD/git_status.txt" 2>/dev/null || true

git diff --binary \
    > "$RECORD/code_changes.patch" 2>/dev/null || true

# 5. 构建训练或测试命令
CMD=(
    torchrun
    --standalone
    --nproc_per_node="$NPROC"
    opengait/main.py
    --cfgs "$CFG"
    --phase "$PHASE"
    --log_to_file
)

if [[ -n "$ITER" ]]; then
    CMD+=(--iter "$ITER")
fi

printf '%q ' "${CMD[@]}" > "$RECORD/command.txt"
printf '\n' >> "$RECORD/command.txt"

# 6. 开始计时
START=$(date +%s)

{
    echo "Run ID: $RUN_ID"
    echo "Phase: $PHASE"
    echo "Config: $CFG"
    echo "Output: $OUT"
    echo "Start: $(date '+%Y-%m-%d %H:%M:%S %z')"
} > "$RECORD/run_summary.txt"

# 正常退出或收到可处理的中断信号时记录耗时
finish() {
    local status="$1"
    local end
    local duration

    end=$(date +%s)
    duration=$((end - START))

    {
        echo "End: $(date '+%Y-%m-%d %H:%M:%S %z')"
        echo "Duration_seconds: $duration"
        printf 'Duration: %02d:%02d:%02d\n' \
            $((duration / 3600)) \
            $(((duration % 3600) / 60)) \
            $((duration % 60))
        echo "Exit_code: $status"
    } >> "$RECORD/run_summary.txt"

    echo "实验记录已写入: $RECORD"
}

trap 'finish $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# 7. 启动 OpenGait，并保存全部终端输出
echo "========== 开始 $PHASE =========="

"${CMD[@]}" 2>&1 | tee "$RECORD/console.log"
