#!/usr/bin/env python3
"""Backward-compatible adapter for the generic augmentation framework."""

import os

from scripts.augmentations.appender import append_augmentation, code_hash
from scripts.augmentations.registry import get_plugin


def append_symmetric_episodes(pool_path, source_seed_file, base_count, target_count,
                              embodiment_name="aloha-agilex", dry_run=False):
    scripts_dir = os.path.dirname(__file__)
    package_dir = os.path.join(scripts_dir, "augmentations")
    hash_paths = [os.path.join(package_dir, name) for name in os.listdir(package_dir)
                  if name.endswith(".py")]
    hash_paths += [os.path.join(package_dir, "plugins", name)
                   for name in os.listdir(os.path.join(package_dir, "plugins"))
                   if name.endswith(".py")]
    hash_paths.append(os.path.join(scripts_dir, "mirror_put_object_cabinet.py"))
    context = {
        "task_name": "put_object_cabinet", "task_config": "demo_clean",
        "embodiment_name": embodiment_name, "code_sha256": code_hash(hash_paths),
    }
    return append_augmentation(
        plugin=get_plugin("put_object_cabinet_symmetry"), pool_path=pool_path,
        source_seed_file=source_seed_file, base_count=base_count,
        target_count=target_count, context=context, dry_run=dry_run,
    )
