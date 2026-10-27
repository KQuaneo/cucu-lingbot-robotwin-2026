from pathlib import Path
import datetime
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tarfile

BASE = Path('/var/tmp/lingbotvla-final-handoff-20261004')
BASE.mkdir(parents=True, exist_ok=True)
os.environ['GIT_LFS_SKIP_SMUDGE'] = '1'
REPOS = {'RoboTwin': Path('/RoboTwin'), 'XPolicyLab': Path('/RoboTwin/XPolicyLab'),
         'lingbot-vla-v2': Path('/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2')}
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()
def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])

entries, repositories, checks = [], {}, []
for name, root in REPOS.items():
    bundle = BASE / (name + '.bundle')
    subprocess.run(['git', '-C', str(root), 'bundle', 'create', str(bundle), 'HEAD'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    patch = BASE / (name + '.working.patch')
    options = ['diff', '--binary', 'HEAD', '--', '.']
    if name == 'RoboTwin': options += [':(exclude)XPolicyLab']
    patch.write_bytes(git(root, *options))
    commit = git(root, 'rev-parse', 'HEAD').decode().strip()
    test = BASE / 'restore_verify' / name
    if not test.exists():
        subprocess.run(['git', 'clone', '--no-checkout', str(bundle), str(test)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    subprocess.run(['git', '-C', str(test), 'checkout', '--force', '--detach', commit], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    # A fresh checkout plus the exact uncommitted patch must be applicable.
    if patch.stat().st_size:
        subprocess.run(['git', '-C', str(test), 'apply', '--check', str(patch)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    subprocess.run(['git', '-C', str(test), 'bundle', 'verify', str(bundle)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    repositories[name] = {'path': str(root), 'commit': commit, 'bundle': name + '.bundle',
                          'patch': name + '.working.patch', 'bundle_verify_passed': True,
                          'clone_checkout_passed': True, 'patch_apply_check_passed': True,
                          'status': git(root, 'status', '--porcelain=v1', '--untracked-files=normal').decode()}
    for path in [bundle, patch]: entries.append((path, 'repositories/' + path.name))

ALLOWED = {'.py', '.sh', '.yaml', '.yml', '.toml', '.json', '.jsonl', '.csv', '.tsv', '.txt', '.log', '.md', '.png', '.svg', '.parquet', '.metadata', '.patch', '.pid'}
PRUNED = {'.git', '__pycache__', 'checkpoints', 'hf_ckpt', 'hf_ckpt_6000', 'eval_cache', 'models', 'source', 'site-packages', '.venv'}
roots = [Path('/workspace/runtime/scripts'), Path('/workspace/runtime/patches'), Path('/workspace/runtime/outputs'),
         Path('/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts'), Path('/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/training')]
roots += [p for p in Path('/var/tmp').glob('robotwin-*') if p.is_dir()]
seen, skipped = set(), []
def include(path):
    if not path.is_file() or path.is_symlink() or path in seen: return
    if path.stat().st_size > 100 * 1024 * 1024:
        skipped.append({'path': str(path), 'reason': 'small-file supplement size limit', 'bytes': path.stat().st_size}); return
    seen.add(path); entries.append((path, 'original_paths/' + str(path).lstrip('/')))
for root in roots:
    if not root.is_dir(): continue
    for parent, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in PRUNED and not d.startswith('global_step_') and not Path(parent, d).is_symlink()]
        for name in files:
            path = Path(parent, name)
            if path.suffix in ALLOWED or name.startswith('events.out.tfevents.') or name in {'exit_code', 'eval_exit_code', 'clean50_exit_code', 'run_start_iso'}:
                include(path)
manifest_path = Path('/workspace/runtime/robotwin_demo_clean_joint_v30.txt')
include(manifest_path)
datasets = []
for text in manifest_path.read_text().splitlines():
    text = text.strip()
    if not text or text.startswith('#'): continue
    path = Path(text.split()[-1])
    if not path.is_dir():
        raise ValueError('Unexpected training data manifest entry: ' + text)
    meta = path / 'meta'
    metadata = []
    for file in sorted(meta.rglob('*')):
        if file.is_file(): include(file); metadata.append(str(file))
    inventory = []
    for parent, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d != '.cache']
        for name in files:
            file = Path(parent, name)
            if file.is_file(): inventory.append({'relative_path': str(file.relative_to(path)), 'bytes': file.stat().st_size})
    datasets.append({'path': str(path), 'metadata_files': metadata, 'data_file_inventory': inventory,
                     'total_bytes': sum(f['bytes'] for f in inventory), 'raw_data_downloaded': False})
model_root = Path('/models/robotwin-persistent/models')
for root in [Path('/models/robotwin-persistent/assets/.cache/huggingface/download'),
             *[p / '.cache/huggingface/download' for p in model_root.iterdir() if p.is_dir()]]:
    if root.exists():
        for file in root.rglob('*.metadata'): include(file)
for file in (model_root / 'Qwen3-VL-4B-Instruct-config-tokenizer').rglob('*'):
    if file.is_file() and file.suffix in ALLOWED: include(file)
for file in [model_root / 'robbyant_lingbot-vla-v2-6b' / 'dino_video/config.yaml',
             model_root / 'robbyant_lingbot-vla-v2-6b' / 'config.json',
             model_root / 'robbyant_lingbot-vla-v2-6b' / 'model.safetensors.index.json',
             model_root / 'robbyant_lingbot-vla-v2-6b-robotwin' / 'lingbotvla_cli.yaml',
             model_root / 'moge-2-vitb-normal' / 'README.md']:
    include(file)

packages = json.loads(subprocess.check_output([sys.executable, '-m', 'pip', 'list', '--format=json'], stderr=subprocess.DEVNULL))
native = subprocess.run(['dpkg-query', '-W', '-f=${Package}\t${Version}\n'], capture_output=True, text=True)
(BASE / 'system_packages.tsv').write_text(native.stdout)
entries.append((BASE / 'system_packages.tsv', 'metadata/system_packages.tsv'))
environment = {'python': sys.version, 'platform': platform.platform(), 'packages': packages,
               'known_environment': {key: os.environ.get(key) for key in ['AITER_TRITON_ONLY', 'FLASH_ATTENTION_TRITON_AMD_ENABLE', 'PYOPENGL_PLATFORM']},
               'complete_environment_image_downloaded': False}
links = []
for path in [Path('/RoboTwin/assets'), Path('/RoboTwin/data'), Path('/RoboTwin/eval_result'), *Path('/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models').glob('*')]:
    if path.is_symlink(): links.append({'path': str(path), 'target': os.readlink(path), 'resolved': str(path.resolve())})
items = [{'archive_path': name, 'original_path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)} for path, name in entries]
manifest = {'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'repositories': repositories,
            'files': items, 'training_datasets': datasets, 'environment': environment, 'symlinks': links,
            'skipped_files': skipped, 'excluded_bulk_resources': ['raw training videos/parquet payloads', 'RoboTwin asset meshes/textures', 'base/official/teacher weights', 'installed Python/ROCm environment'],
            'source_restore_verified': True, 'full_runtime_restore_smoke_performed': False}
payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
(BASE / 'manifest.json').write_bytes(payload)
archive = BASE / 'source_and_handoff.tar.gz'
with tarfile.open(archive, 'w:gz') as handle:
    for path, name in entries: handle.add(path, arcname=name, recursive=False)
    info = tarfile.TarInfo('metadata/manifest.json'); info.size = len(payload); handle.addfile(info, io.BytesIO(payload))
result = {'remote_archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest(archive),
          'file_count': len(entries), 'dataset_count': len(datasets), 'dataset_metadata_files': sum(len(d['metadata_files']) for d in datasets),
          'source_restore_verified': True, 'manifest': str(BASE / 'manifest.json')}
(BASE / 'transfer.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result))
