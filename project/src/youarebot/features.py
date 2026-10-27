"""Признаки поверх таблицы участников: позиция во времени, стиль, эмбеддинги, kNN по репликам, таблица сообщений."""
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------- позиция в файле (порядок json ~ время) ----------


def add_rel_pos(df: pd.DataFrame) -> pd.Series:
    """Относительная позиция диалога внутри своей выборки (0..1)."""
    order = pd.Series(np.arange(df.dialog_id.nunique()), index=df.dialog_id.unique())
    return (df.dialog_id.map(order) + 0.5) / len(order)


def position_features(ref: pd.DataFrame, query: pd.DataFrame, bandwidths, prior_rate, prior_weight):
    """Сглаженная по времени доля ботов среди ref-диалогов рядом с позицией query."""
    out = {}
    rp, qp = ref.rel_pos.values, query.rel_pos.values
    same_role = query.participant_index.values[:, None] == ref.participant_index.values[None, :]
    for bw in bandwidths:
        w = np.exp(-0.5 * ((qp[:, None] - rp[None, :]) / bw) ** 2)
        ws = w * same_role
        out[f"pos_role_{bw}"] = (ws @ ref.y.values + prior_rate * prior_weight) / (ws.sum(1) + prior_weight)
        out[f"pos_any_{bw}"] = (w @ ref.y.values + prior_rate * prior_weight) / (w.sum(1) + prior_weight)
        out[f"pos_dens_{bw}"] = w.sum(1) / len(rp)
    return pd.DataFrame(out, index=query.index)


# ---------- стиль ----------

STYLE_PATTERNS = {
    "emdash": "—", "endash": "–", "yo": "[ёЁ]", "guil": "[«»]", "ellips": "…|\\.\\.\\.", "smile": "\\)\\)|\\(\\(",
    "curly": "[’“”]", "comma": ",", "colon_sp": " :", "lower_start": "^[a-zа-яё]", "dot_end": "\\.$",
    "excl2": "!!", "q_ru": "\\?$", "latin_word": "\\b[a-zA-Z]{3,}\\b", "caps_word": "\\b[A-ZА-ЯЁ]{3,}\\b",
}


def style_table(dialogs: dict) -> pd.DataFrame:
    rows = {}
    for d, ms in dialogs.items():
        for p in (0, 1):
            mine = [m["text"].strip() for m in ms if m["participant_index"] == p]
            n = max(len(mine), 1)
            r = {f"sty_{k}": sum(bool(re.search(v, t)) for t in mine) / n for k, v in STYLE_PATTERNS.items()}
            r["sty_chars"] = sum(len(t) for t in mine)
            rows[(d, p)] = r
    return pd.DataFrame.from_dict(rows, orient="index")


def join_by_participant(table: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    k = list(zip(d.dialog_id, d.participant_index))
    return d.join(pd.DataFrame(table.loc[k].values, columns=table.columns, index=d.index))


# ---------- эмбеддинги реплик ----------


def short_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def compute_embeddings(dialogs: dict, model_name: str, cache_dir, device, batch_size) -> dict:
    """{текст реплики: вектор}. Кэшируется в cache_dir/emb_<model>.pkl."""
    path = Path(cache_dir) / f"emb_{short_name(model_name)}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    import torch
    from sentence_transformers import SentenceTransformer

    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    texts = sorted({m["text"] for ms in dialogs.values() for m in ms})
    prefix = "query: " if "e5" in model_name else ""
    model = SentenceTransformer(model_name, device=device)
    E = model.encode([prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                     show_progress_bar=False)
    emb = dict(zip(texts, E))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(emb, f)
    return emb


def participant_embeddings(dialogs: dict, E: dict):
    """X: [средний вектор своих реплик | средний вектор реплик собеседника]; S: сходство реплик."""
    dim = len(next(iter(E.values())))
    rows, extras, keys = [], [], []
    for did, ms in dialogs.items():
        for p in (0, 1):
            mine = [E[m["text"]] for m in ms if m["participant_index"] == p]
            oth = [E[m["text"]] for m in ms if m["participant_index"] != p]
            sims = [float(E[ms[i - 1]["text"]] @ E[m["text"]]) for i, m in enumerate(ms)
                    if i and m["participant_index"] == p and ms[i - 1]["participant_index"] != p]
            own = [float(a @ b) for a, b in zip(mine, mine[1:])]
            rows.append(np.r_[np.mean(mine, 0) if mine else np.zeros(dim), np.mean(oth, 0) if oth else np.zeros(dim)])
            extras.append(dict(sim_mean=np.mean(sims) if sims else 0, sim_max=max(sims) if sims else 0,
                               sim_min=min(sims) if sims else 0, own_sim=np.mean(own) if own else 0))
            keys.append((did, p))
    idx = pd.MultiIndex.from_tuples(keys, names=["dialog_id", "participant_index"])
    return pd.DataFrame(np.array(rows), index=idx), pd.DataFrame(extras, index=idx)


# ---------- kNN по репликам ----------


def _msg_table(dialogs: dict, dialog_ids) -> pd.DataFrame:
    rows = []
    for d in dialog_ids:
        ms = dialogs[d]
        for i, m in enumerate(ms):
            prev = ms[i - 1]["text"] if i and ms[i - 1]["participant_index"] != m["participant_index"] else ""
            rows.append((d, m["participant_index"], m["text"], prev))
    return pd.DataFrame(rows, columns=["dialog_id", "participant_index", "text", "prev"])


def knn_features(dialogs, E, ref, query, ks, temperature):
    """Доля ботов среди авторов ближайших реплик (m) и пар «реплика собеседника → ответ» (p) из ref."""
    lab = dict(zip(zip(ref.dialog_id, ref.participant_index), ref.y))
    R, Q = _msg_table(dialogs, ref.dialog_id.unique()), _msg_table(dialogs, query.dialog_id.unique())
    R["y"] = [lab[k] for k in zip(R.dialog_id, R.participant_index)]
    ER, EQ = np.stack([E[t] for t in R.text]), np.stack([E[t] for t in Q.text])
    z = np.zeros(ER.shape[1])
    PR = np.hstack([ER, np.stack([E[t] if t else z for t in R.prev])]) / np.sqrt(2)
    PQ = np.hstack([EQ, np.stack([E[t] if t else z for t in Q.prev])]) / np.sqrt(2)
    out = {}
    for name, A, B in (("m", EQ, ER), ("p", PQ, PR)):
        S = A @ B.T
        order = np.argsort(-S, axis=1)[:, :max(ks)]
        ys, ss = R.y.values[order], np.take_along_axis(S, order, 1)
        for k in ks:
            w = np.exp(temperature * ss[:, :k])
            out[f"k{name}_y{k}"] = (ys[:, :k] * w).sum(1) / w.sum(1)
        out[f"k{name}_s1"] = ss[:, 0]
    F = pd.concat([Q[["dialog_id", "participant_index"]], pd.DataFrame(out)], axis=1)
    agg = F.groupby(["dialog_id", "participant_index"]).agg(["mean", "max", "min"])
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg


# ---------- модель на уровне сообщений ----------


class MessageTable:
    """Все сообщения (train+test) с контекстом и TF-IDF-матрицей для логрега на уровне сообщений."""

    def __init__(self, dialogs: dict, c: dict):
        rows = []
        for d, ms in dialogs.items():
            for i, m in enumerate(ms):
                p = m["participant_index"]
                rows.append(dict(
                    dialog_id=d, participant_index=p, text=m["text"],
                    prev=ms[i - 1]["text"] if i and ms[i - 1]["participant_index"] != p else "",
                    idx=sum(1 for x in ms[:i] if x["participant_index"] == p),
                    last_msg=int(all(x["participant_index"] != p for x in ms[i + 1:])),
                    is_final=int(i == len(ms) - 1),
                ))
        M = pd.DataFrame(rows)
        texts = M.text.tolist()
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(c["char_ngram"]), min_df=c["char_min_df"],
                             sublinear_tf=True, lowercase=False).fit(texts)
        vw = TfidfVectorizer(ngram_range=tuple(c["word_ngram"]), sublinear_tf=True).fit(texts)
        vp = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(c["prev_char_ngram"]), min_df=c["prev_min_df"],
                             sublinear_tf=True).fit(texts)
        meta = np.c_[M.idx.clip(0, 5) / 5, M.last_msg, M.is_final, M.idx == 0, M.prev == ""].astype(float)
        self.X = sp.hstack([vc.transform(M.text), vw.transform(M.text), c["prev_weight"] * vp.transform(M.prev),
                            sp.csr_matrix(meta)]).tocsr()
        self.M = M
        self.key = pd.MultiIndex.from_frame(M[["dialog_id", "participant_index"]])
