import re
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


TRAIN_LOG = Path("/workspace/runtime/outputs/logs/lora_10000steps_8gpu.log")
TB_DIR = Path("/workspace/runtime/outputs/tensorboard/lora_10000steps_8gpu")
STATE_FILE = TB_DIR / "last_step.txt"
MAX_STEPS = 10_000

STEP_PATTERN = re.compile(r"Step\s+(\d+)/\d+,\s+Epoch\s+(\d+),\s+(.*)")
METRIC_PATTERN = re.compile(
    r"(Loss|VLA_Loss|Depth_Loss|Future_Depth_Loss|"
    r"FutureVideo_Loss|SeqWise_Loss|RouterZ_Loss|"
    r"MaxVio|AvgSigmoid|GradNorm|LR|StepTime)"
    r"\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
METRIC_NAMES = {
    "Loss": "train/loss",
    "VLA_Loss": "train/vla_loss",
    "Depth_Loss": "train/depth_loss",
    "Future_Depth_Loss": "train/future_depth_loss",
    "FutureVideo_Loss": "train/future_video_loss",
    "SeqWise_Loss": "train/seqwise_loss",
    "RouterZ_Loss": "train/router_z_loss",
    "MaxVio": "train/max_vio",
    "AvgSigmoid": "train/avg_sigmoid",
    "GradNorm": "train/grad_norm",
    "LR": "train/learning_rate",
    "StepTime": "performance/step_time_seconds",
}


def load_last_step() -> int:
    if not STATE_FILE.exists():
        return 0
    try:
        return int(STATE_FILE.read_text().strip())
    except ValueError:
        return 0


def collect_records(last_step: int) -> dict[int, dict[str, float]]:
    text = TRAIN_LOG.read_text(errors="replace").replace("\r", "\n")
    records: dict[int, dict[str, float]] = {}
    for line in text.splitlines():
        match = STEP_PATTERN.search(line)
        if not match:
            continue
        step = int(match.group(1))
        if step <= last_step or step > MAX_STEPS:
            continue
        metrics = {"train/epoch": float(match.group(2))}
        for key, value in METRIC_PATTERN.findall(match.group(3)):
            metrics[METRIC_NAMES[key]] = float(value)
        records[step] = metrics
    return records


def main() -> None:
    TB_DIR.mkdir(parents=True, exist_ok=True)
    last_step = load_last_step()
    writer = SummaryWriter(log_dir=str(TB_DIR))
    try:
        while True:
            if not TRAIN_LOG.exists():
                print(f"Waiting for {TRAIN_LOG}", flush=True)
                time.sleep(10)
                continue
            records = collect_records(last_step)
            for step in sorted(records):
                for name, value in records[step].items():
                    writer.add_scalar(name, value, step)
                last_step = step
                STATE_FILE.write_text(str(last_step))
                print(f"Wrote TensorBoard scalars for step {step}", flush=True)
            writer.flush()
            if last_step >= MAX_STEPS:
                print("Reached step 10000; converter exiting.", flush=True)
                break
            time.sleep(10)
    finally:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
