import os
import shutil
from pathlib import Path
import re

workdir = Path('/home/yzy/mmocr/workdir_det')
archive_dir = workdir / 'archive'
grid_search_dest = workdir / 'experiments' / 'grid_search'

archive_dir.mkdir(parents=True, exist_ok=True)
grid_search_dest.mkdir(parents=True, exist_ok=True)

print(f"Cleaning {workdir}...")

# 1. Move grid_search_* directories
for item in workdir.iterdir():
    if item.is_dir() and item.name.startswith('grid_search_'):
        print(f"Moving {item.name} to {grid_search_dest}")
        shutil.move(str(item), str(grid_search_dest / item.name))

# 2. Archive timestamped directories in workdir_det root
# Pattern: YYYYMMDD_HHMMSS
timestamp_pattern = re.compile(r'^\d{8}_\d{6}$')

for item in workdir.iterdir():
    if item.is_dir() and timestamp_pattern.match(item.name):
        print(f"Archiving {item.name} to {archive_dir}")
        shutil.move(str(item), str(archive_dir / item.name))

# 3. Archive timestamped directories inside model folders (dbnet, fcenet)
model_dirs = ['dbnet', 'fcenet']
for model_dir_name in model_dirs:
    model_path = workdir / model_dir_name
    if model_path.exists() and model_path.is_dir():
        print(f"Cleaning model dir: {model_path}")
        model_archive = model_path / 'archive'
        model_archive.mkdir(exist_ok=True)
        
        for item in model_path.iterdir():
            if item.is_dir() and timestamp_pattern.match(item.name):
                print(f"  Archiving run {item.name} to {model_archive}")
                shutil.move(str(item), str(model_archive / item.name))

print("Deep cleanup complete.")
