#!/usr/bin/env python3
"""Install, inspect, or roll back the augmentation framework on RoboTwin."""

import argparse
import datetime
import hashlib
import json
import os
import shutil


def files_under(root):
    result = []
    for directory, _, files in os.walk(root):
        for name in files:
            result.append(os.path.relpath(os.path.join(directory, name), root))
    return sorted(result)


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def install(bundle, root, dry_run):
    source_root = os.path.join(bundle, "RoboTwin")
    if not os.path.isfile(os.path.join(root, "scripts", "collect_data.py")):
        raise RuntimeError(f"not a compatible RoboTwin root: {root}")
    files = files_under(source_root)
    plan = [{"path": rel, "sha256": digest(os.path.join(source_root, rel)),
             "replaces_existing": os.path.exists(os.path.join(root, rel))} for rel in files]
    print(json.dumps({"root": root, "dry_run": dry_run, "files": plan}, indent=2))
    if dry_run:
        return
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(root, f".augmentation_backup_{stamp}")
    manifest = {"root": root, "files": {}}
    for rel in files:
        src, dst = os.path.join(source_root, rel), os.path.join(root, rel)
        existed = os.path.exists(dst)
        manifest["files"][rel] = {"existed": existed}
        if existed:
            saved = os.path.join(backup, "files", rel)
            os.makedirs(os.path.dirname(saved), exist_ok=True)
            shutil.copy2(dst, saved)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    os.makedirs(backup, exist_ok=True)
    with open(os.path.join(backup, "manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    with open(os.path.join(root, ".augmentation_last_backup"), "w", encoding="utf-8") as stream:
        stream.write(backup + "\n")
    print(f"installed; rollback backup: {backup}")


def rollback(root, backup):
    with open(os.path.join(backup, "manifest.json"), encoding="utf-8") as stream:
        manifest = json.load(stream)
    if os.path.abspath(manifest["root"]) != os.path.abspath(root):
        raise RuntimeError("backup belongs to another RoboTwin root")
    for rel, meta in manifest["files"].items():
        dst = os.path.join(root, rel)
        if meta["existed"]:
            src = os.path.join(backup, "files", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        elif os.path.exists(dst):
            os.remove(dst)
    print(f"rolled back from {backup}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/RoboTwin")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--rollback")
    args = parser.parse_args()
    if sum((args.dry_run, args.install, bool(args.rollback))) != 1:
        parser.error("choose exactly one of --dry-run, --install, --rollback BACKUP")
    root = os.path.abspath(args.root)
    if args.rollback:
        rollback(root, os.path.abspath(args.rollback))
    else:
        install(os.path.dirname(os.path.abspath(__file__)), root, args.dry_run)


if __name__ == "__main__":
    main()
