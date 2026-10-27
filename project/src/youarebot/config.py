"""Загрузка конфига и ограничение ресурсов."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # project/


def limit_threads(n: int) -> None:
    """Вызывать до импорта numpy/sklearn/torch: ограничивает потоки BLAS/OpenMP."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_config(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path) -> Path:
    """Путь из конфига: относительный считается от папки project/."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def flatten(d: dict, prefix: str = "") -> dict:
    """{'a': {'b': 1}} -> {'a.b': 1} для mlflow.log_params."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out
