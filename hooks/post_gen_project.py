import os
import shutil

STARTING_POINT = "{{ cookiecutter.starting_point }}"
STARTERS_DIR = os.path.join("src", "_starters")

if STARTING_POINT != "template":
    starter_path = os.path.join(STARTERS_DIR, STARTING_POINT)
    for filename in ("models.py", "datasets.py", "configuration.toml"):
        src_file = os.path.join(starter_path, filename)
        if os.path.exists(src_file):
            shutil.copy2(src_file, os.path.join("src", filename))

if os.path.exists(STARTERS_DIR):
    shutil.rmtree(STARTERS_DIR)
