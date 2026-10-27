"""End-to-end эксперимент: данные -> признаки -> CV -> обучение на всём train -> сабмит. Всё логируется в MLflow.

python -m youarebot.pipeline --config configs/config.yaml [--no-cv] [--set level2.n_seeds=3 ...]
"""
import argparse
import hashlib
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import yaml

from .config import ROOT, flatten, limit_threads, load_config, resolve


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--no-cv", action="store_true", help="пропустить CV (только обучение и сабмит)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="переопределить параметр конфига, например level2.n_seeds=3")
    return ap.parse_args()


def apply_overrides(cfg, items):
    for item in items:
        key, value = item.split("=", 1)
        node = cfg
        *path, last = key.split(".")
        for p in path:
            node = node[p]
        node[last] = yaml.safe_load(value)
    return cfg


def resolve_tracking_uri(uri: str) -> str:
    """Относительный sqlite-путь считаем от папки project/, чтобы база не зависела от cwd."""
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not uri.startswith(prefix + "/"):
        return prefix + str(ROOT / uri[len(prefix):])
    return uri


def file_md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    if args.no_cv:
        cfg["cv"]["enabled"] = False
    limit_threads(cfg["resources"]["threads"])

    # тяжёлые импорты — после ограничения потоков
    import mlflow
    import numpy as np
    import pandas as pd
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold

    from .data import build_tables, load_raw
    from .features import MessageTable, compute_embeddings, participant_embeddings, short_name, style_table
    from .models import TextLR, run_stack

    t0 = time.time()
    mlflow.set_tracking_uri(resolve_tracking_uri(cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    with mlflow.start_run(run_name=cfg["run_name"]) as run:
        mlflow.log_params(flatten(cfg))
        mlflow.set_tags({
            "mlflow.source.name": "src/youarebot/pipeline.py",
            "python": platform.python_version(),
            "machine": platform.machine(),
            **{f"data_md5.{f}": file_md5(resolve(cfg["data"]["dir"]) / f)
               for f in ("train.json", "ytrain.csv", "test.json", "ytest.csv")},
        })

        # --- данные и признаки, не зависящие от меток ---
        raw = load_raw(resolve(cfg["data"]["dir"]))
        dtr, dte = build_tables(raw)
        dialogs = raw.dialogs
        log(f"данные: {len(dtr)} участников train, {len(dte)} test")
        r, ce = cfg["resources"], cfg["level1"]["embeddings"]
        embs = {m: compute_embeddings(dialogs, m, resolve(cfg["data"]["cache_dir"]), r["embed_device"],
                                      r["embed_batch_size"])
                for m in sorted(set(ce["models"]) | {ce["sim_model"], cfg["level1"]["knn"]["model"]})}
        emb_X = {m: participant_embeddings(dialogs, embs[m])[0] for m in ce["models"]}
        emb_S = participant_embeddings(dialogs, embs[ce["sim_model"]])[1]
        ctx = dict(dialogs=dialogs, emb_X=emb_X, emb_S=emb_S, E_knn=embs[cfg["level1"]["knn"]["model"]],
                   style=style_table(dialogs),
                   messages=MessageTable(dialogs, cfg["level1"]["message_lr"]),
                   text_lr=TextLR(pd.concat([dtr.text_mine, dte.text_mine]).values, cfg))
        log(f"признаки без меток готовы ({time.time() - t0:.0f}s)")

        # --- CV: внешние фолды по диалогам, внутри — полный пайплайн 1-го и 2-го уровня ---
        if cfg["cv"]["enabled"]:
            cls = dtr.groupby("dialog_id", sort=False).y.apply(lambda s: int(s.values @ [1, 2]))
            dlg = cls.index.values
            scores, oof_all = [], []
            for seed in cfg["cv"]["seeds"]:
                oof = np.zeros(len(dtr))
                skf = StratifiedKFold(cfg["cv"]["n_splits"], shuffle=True, random_state=seed)
                for k, (a, b) in enumerate(skf.split(dlg, cls.values)):
                    va_mask = dtr.dialog_id.isin(set(dlg[b])).values
                    p, _ = run_stack(dtr[~va_mask], dtr[va_mask].assign(y=0), ctx, cfg)
                    oof[va_mask] = p
                    fold_ll = log_loss(dtr.y.values[va_mask], p)
                    mlflow.log_metric(f"cv_fold_logloss_seed{seed}", fold_ll, step=k)
                    log(f"seed {seed} fold {k}: {fold_ll:.4f} ({time.time() - t0:.0f}s)")
                s = log_loss(dtr.y.values, oof)
                scores.append(s)
                mlflow.log_metric("cv_logloss_by_seed", s, step=seed)
                oof_all.append(pd.DataFrame({"dialog_id": dtr.dialog_id, "participant_index": dtr.participant_index,
                                             "y": dtr.y, "oof": oof, "cv_seed": seed}))
            mlflow.log_metric("cv_logloss", float(np.mean(scores)))
            mlflow.log_metric("cv_logloss_std", float(np.std(scores)))
            log(f"CV LogLoss: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
            _log_df(mlflow, pd.concat(oof_all), "oof_predictions.csv")

        # --- обучение на всём train и предсказание теста ---
        te = dte.drop(columns="ID").assign(y=0)
        pred, imp = run_stack(dtr, te, ctx, cfg)
        ids = te.dialog_id + "_" + te.participant_index.astype(str)
        sub = raw.ytest[["ID"]].copy()
        sub["is_bot"] = sub.ID.map(dict(zip(ids, pred)))
        assert len(sub) == len(raw.ytest) and sub.is_bot.notna().all(), "в сабмите не хватает ID"
        out = resolve(cfg["output"]["submission"])
        out.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out, index=False)

        mlflow.log_metric("n_features", len(imp))
        mlflow.log_metric("runtime_sec", time.time() - t0)
        mlflow.log_artifact(str(out))
        _log_df(mlflow, imp.rename("gain").reset_index().rename(columns={"index": "feature"}), "feature_importance.csv")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "config_resolved.yaml"
            p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
            mlflow.log_artifact(str(p))
        mlflow.log_artifact(str(ROOT / "requirements.txt"))
        log(f"сабмит: {out} | run_id: {run.info.run_id} | {time.time() - t0:.0f}s")


def _log_df(mlflow, df, name):
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, name)
        df.to_csv(p, index=False)
        mlflow.log_artifact(p)


def log(msg):
    print(f"[youarebot] {msg}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
