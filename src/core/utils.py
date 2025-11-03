import json
import os
from pathlib import Path
from typing import Any, Dict


def safe_write_json(data: Dict[str, Any], out_path: Path) -> None:

    tmp_path = str(out_path) + ".tmp"

    # ensure parent dir exists
    os.makedirs(out_path.parent, exist_ok=True)

    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=4)

    os.replace(tmp_path, out_path)


def is_valid_json(path: Path) -> bool:
    # Validate given path is a valid json file

    if not path.exists() or path.stat().st_size < 5:
        return False
    try:
        with open(path, "r", encoding="utf-8") as fh:
            json.load(fh)
        return True
    except Exception:
        return False
