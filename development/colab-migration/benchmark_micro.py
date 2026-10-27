"""Run sequential, isolated FP32 Full-SFT batch-256 throughput checks in Colab."""
import json, os, re, subprocess, sys, time
from pathlib import Path
import yaml

D = Path('/content/drive/MyDrive/LingBot-Colab-Stage2-20261008')
L = Path('/content/lingbot-stage2')
P = L / 'package'
T = P / 'lingbot/tasks/vla/train_lingbotvla.py'
run = D / 'runs' / ('micro_benchmark_' + time.strftime('%Y%m%d_%H%M%S', time.gmtime()))
run.mkdir(parents=True)
def status(phase, **kw):
    value = dict(phase=phase, time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 precision='FP32', use_lora=False, global_batch_size=256,
                 formal_training_started=False, benchmark_directory=str(run), **kw)
    (D/'micro_benchmark_status.json').write_text(json.dumps(value, indent=2))
    print(json.dumps(value), flush=True)

s = T.read_text()
if 'LINGBOT_MICRO_BENCHMARK' not in s:
    (run/'train_lingbotvla_before_benchmark.py').write_text(s)
    old = '            for micro_batch in micro_batches:\n'
    assert s.count(old) == 1
    s = s.replace(old, '            for bench_micro_index, micro_batch in enumerate(micro_batches, 1):\n')
    old = '                del micro_batch\n'
    assert s.count(old) == 1
    s = s.replace(old, '''                if os.environ.get("LINGBOT_MICRO_BENCHMARK"):
                    torch.cuda.synchronize()
                    elapsed = time.time() - start_time
                    progress = dict(micro_batch=args.train.micro_batch_size, completed_micro_batches=bench_micro_index,
                                    total_micro_batches=len(micro_batches), elapsed_seconds=elapsed,
                                    samples_per_second=bench_micro_index * args.train.micro_batch_size / max(elapsed, 1e-6),
                                    loss=float(loss.detach()) * len(micro_batches),
                                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                    import json as bench_json
                    from pathlib import Path as BenchPath
                    BenchPath(os.environ["LINGBOT_MICRO_PROGRESS"]).write_text(bench_json.dumps(progress, indent=2))
                    if bench_micro_index == 1 or bench_micro_index % 8 == 0 or bench_micro_index == len(micro_batches):
                        print("MICRO_PROGRESS " + bench_json.dumps(progress), flush=True)
                del micro_batch
''')
    old = '            already_saved = last_checkpoint_step == global_step'
    assert old in s
    s = s.replace(old, '            already_saved = last_checkpoint_step == global_step or bool(os.environ.get("LINGBOT_MICRO_BENCHMARK"))')
    compile(s, str(T), 'exec')
    T.write_text(s)

base = yaml.safe_load((L/'train.yaml').read_text())
assert base['train']['enable_fp32'] and not base['train']['use_lora']
results = []
try:
    for micro, accumulation in [(2,128),(4,64)]:
        c = yaml.safe_load(yaml.safe_dump(base))
        out = run / f'micro{micro}'
        c['train'].update(micro_batch_size=micro, gradient_accumulation_steps=accumulation,
                          global_batch_size=256, max_steps=1, save_steps=0, save_epochs=0,
                          output_dir=str(out), enable_resume=False, load_checkpoint_path=None)
        config = L/f'benchmark_micro{micro}.yaml'
        config.write_text(yaml.safe_dump(c, sort_keys=False))
        env = os.environ.copy()
        for k in ['HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','AITER_TRITON_ONLY','FLASH_ATTENTION_TRITON_AMD_ENABLE',
                  'LINGBOT_TRAIN_DEADLINE_UNIX','LINGBOT_CHECKPOINT_INTERVAL_SECONDS']:
            env.pop(k, None)
        progress_file = run/f'micro{micro}_progress.json'
        env.update(CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1', LINGBOT_FP32_TRAIN='1',
                   MALLOC_ARENA_MAX='2', LINGBOT_MICRO_BENCHMARK='1', LINGBOT_MICRO_PROGRESS=str(progress_file),
                   QWEN3VL_PATH=str(P/'qwen'),
                   PYTHONPATH=str(P/'lingbot/lingbotvla/models/vla/vision_models/morgbd_clean/3rd/utils3d'))
        log = run/f'micro{micro}.log'
        status('benchmark_running', micro_batch=micro, accumulation=accumulation, log=str(log))
        command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=1',
                   '-m','tasks.vla.train_lingbotvla',str(config)]
        started = time.monotonic()
        with log.open('w') as f:
            process = subprocess.Popen(command, cwd=P/'lingbot', env=env, stdout=f, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=14400)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=90)
                raise RuntimeError(f'micro{micro}: exceeded four-hour benchmark limit')
        text = log.read_text()
        assert code == 0, f'micro{micro} failed, exit={code}; see {log}'
        assert 'Full-SFT parameters:' in text and re.search(r'Step 1/', text), 'Missing complete optimizer step'
        assert 'Non-finite' not in text and 'OutOfMemoryError' not in text
        progress = json.loads(progress_file.read_text())
        assert progress['completed_micro_batches'] == accumulation
        item = dict(micro_batch=micro, accumulation=accumulation, global_batch_size=256,
                    complete_optimizer_step=True, total_run_seconds=time.monotonic()-started, progress=progress,
                    step_log=re.findall(r'Step 1/[^\n]+', text)[-1], log=str(log))
        results.append(item)
        (run/'results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(item), flush=True)
    best = max(results, key=lambda r:r['progress']['samples_per_second'])
    status('benchmark_complete', results=results, recommended_micro_batch=best['micro_batch'])
except Exception as e:
    status('benchmark_failed', error=str(e), completed_results=results)
    raise
