"""在服务器的 robotwin Python 环境运行，stdout 输出训练日志/CSV/PNG 压缩包。"""
import csv, re, sys, tarfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

log = Path('/workspace/runtime/outputs/logs/full_sft_8gpu_6000steps_20261002.log')
text = re.sub(r'\x1b\[[0-9;]*m', '', log.read_text(errors='replace'))
rows = []
for line in text.splitlines():
    match = re.search(r' - Step (\d+)/\d+, Epoch (\d+), (.*)', line)
    if match:
        values = {k: float(v) for k, v in re.findall(r'([A-Za-z_]+)\s+([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)', match[3])}
        rows.append({'step': int(match[1]), 'epoch': int(match[2]), **values})

base = Path('/var/tmp/robotwin-full-sft-6k-20261002')
csv_path = base / 'backup_training_metrics_final.csv'
fields = ['step', 'epoch'] + sorted(set().union(*(r.keys() for r in rows)) - {'step', 'epoch'})
with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

steps = np.array([r['step'] for r in rows])
fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
for key, color in [('Loss', 'tab:blue'), ('VLA_Loss', 'tab:orange')]:
    y = np.array([r[key] for r in rows])
    axes[0].plot(steps, y, color=color, alpha=.18, lw=.45)
    axes[0].plot(steps[99:], np.convolve(y, np.ones(100)/100, 'valid'), color=color, label=key + ' (100-step mean)')
axes[0].set_yscale('log')
axes[0].set_ylabel('Training loss')
axes[0].legend()
axes[0].set_title('LingBot-VLA-v2 Full-SFT, 6000 steps, 8 GPUs; all logged steps')
for key in ['Depth_Loss', 'Future_Depth_Loss', 'FutureVideo_Loss']:
    y = np.array([r[key] for r in rows])
    axes[1].plot(steps[99:], np.convolve(y, np.ones(100)/100, 'valid'), label=key)
axes[1].set_yscale('log')
axes[1].set_ylabel('Raw teacher losses\n100-step mean')
axes[1].legend()
for key in ['LR', 'Expert_LR']:
    axes[2].plot(steps, [r[key] for r in rows], label=key)
axes[2].set_ylabel('Learning rate')
axes[2].set_xlabel('Training step')
axes[2].legend()
for ax in axes:
    ax.grid(alpha=.2)
fig.tight_layout()
png_path = base / 'backup_loss_curve_final.png'
fig.savefig(png_path, dpi=160)
plt.close(fig)
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
    archive.add(log, arcname=log.name)
    archive.add(csv_path, arcname='training_metrics_6000steps.csv')
    archive.add(png_path, arcname='loss_curve_6000steps.png')
