"""Стекинг. 1-й уровень (OOF-признаки): текстовый логрег, логреги на эмбеддингах, kNN, позиция, логрег по сообщениям.
2-й уровень: LightGBM по участнику + 3-классовый LightGBM по диалогу; бэггинг по сидам и температура."""
import copy

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from .features import (add_rel_pos, join_by_participant, knn_features, position_features, short_name)

EXCLUDE = {"dialog_id", "participant_index", "y"}


class TextLR:
    """LogReg на TF-IDF слов и символьных n-грамм текста участника. Словарь — по всем текстам (без меток)."""

    def __init__(self, all_texts, cfg):
        c = cfg["level1"]["text_lr"]
        self.C = c["C"]
        self.max_iter = cfg["level1"]["lr_max_iter"]
        self.vw = TfidfVectorizer(ngram_range=tuple(c["word_ngram"]), sublinear_tf=True).fit(all_texts)
        self.vc = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(c["char_ngram"]), min_df=c["char_min_df"],
                                  sublinear_tf=True, lowercase=False).fit(all_texts)

    def X(self, d):
        return sp.hstack([self.vw.transform(d.text_mine), self.vc.transform(d.text_mine)]).tocsr()

    def fit_predict(self, a, b):
        m = LogisticRegression(C=self.C, max_iter=self.max_iter).fit(self.X(a), a.y.values)
        return m.predict_proba(self.X(b))[:, 1]


def add_partner(df, cols):
    p = df[["dialog_id", "participant_index"] + cols].copy()
    p["participant_index"] = 1 - p["participant_index"]
    p = p.rename(columns={c: "pt_" + c for c in cols})
    return df.merge(p, on=["dialog_id", "participant_index"], how="left")


def _emb_lr(X, tr, te, C, n_inner, seed, max_iter):
    key = lambda d: list(zip(d.dialog_id, d.participant_index))
    Xtr, Xte = X.loc[key(tr)].values, X.loc[key(te)].values
    oof = np.zeros(len(tr))
    for a, b in GroupKFold(n_inner, shuffle=True, random_state=seed).split(Xtr, tr.y, tr.dialog_id):
        oof[b] = LogisticRegression(C=C, max_iter=max_iter).fit(Xtr[a], tr.y.values[a]).predict_proba(Xtr[b])[:, 1]
    pte = LogisticRegression(C=C, max_iter=max_iter).fit(Xtr, tr.y.values).predict_proba(Xte)[:, 1]
    return oof, pte


def level1(tr, te, ctx, cfg):
    """OOF-признаки для tr (внутренняя CV по диалогам) и признаки для te (модели на всём tr).

    ctx: dialogs, text_lr, emb_X {model: X}, emb_S, E_knn, style, messages.
    """
    c1, seed = cfg["level1"], cfg["seed"]
    n_inner = c1["n_inner"]
    off = c1["seed_offsets"]  # разные сиды внутренних разбиений для разных моделей 1-го уровня
    max_iter = c1["lr_max_iter"]
    pos = c1["position"]
    tr, te = tr.copy(), te.copy()

    # текстовый логрег + позиция во времени
    tr["lr"] = 0.0
    pf = []
    for a, b in GroupKFold(n_inner, shuffle=True, random_state=seed + off["text_lr"]).split(tr, tr.y, tr.dialog_id):
        A, B = tr.iloc[a].copy(), tr.iloc[b].copy()
        tr.iloc[b, tr.columns.get_loc("lr")] = ctx["text_lr"].fit_predict(A, B)
        A["rel_pos"], B["rel_pos"] = add_rel_pos(A), add_rel_pos(B)  # позиция внутри части — как у теста
        pf.append(position_features(A, B, pos["bandwidths"], pos["prior_rate"], pos["prior_weight"]))
    tr = tr.join(pd.concat(pf))
    te["lr"] = ctx["text_lr"].fit_predict(tr, te)
    R = tr.copy()
    R["rel_pos"] = add_rel_pos(R)
    te["rel_pos"] = add_rel_pos(te)
    te = te.join(position_features(R, te, pos["bandwidths"], pos["prior_rate"], pos["prior_weight"]))
    tr["rel_pos"] = add_rel_pos(tr)

    # признаки собеседника
    tr, te = add_partner(tr, c1["partner_features"]), add_partner(te, c1["partner_features"])

    # логреги на эмбеддингах
    for name, X in ctx["emb_X"].items():
        col = "e_" + short_name(name)
        tr[col], te[col] = _emb_lr(X, tr, te, c1["embeddings"]["lr_C"], n_inner, seed + off["embeddings"], max_iter)
        for d in (tr, te):
            d["pt_" + col] = d.groupby("dialog_id")[col].transform("sum") - d[col]
    for d in (tr, te):
        k = list(zip(d.dialog_id, d.participant_index))
        for c in ctx["emb_S"].columns:
            d[c] = ctx["emb_S"].loc[k, c].values

    # kNN по репликам
    kc = c1["knn"]
    parts = []
    for a, b in GroupKFold(n_inner, shuffle=True, random_state=seed + off["knn"]).split(tr, tr.y, tr.dialog_id):
        parts.append(knn_features(ctx["dialogs"], ctx["E_knn"], tr.iloc[a], tr.iloc[b], kc["ks"], kc["temperature"]))
    j = lambda d, K: d.merge(K, left_on=["dialog_id", "participant_index"], right_index=True, how="left")
    tr = j(tr, pd.concat(parts))
    te = j(te, knn_features(ctx["dialogs"], ctx["E_knn"], tr, te, kc["ks"], kc["temperature"]))

    # логрег на уровне сообщений, агрегированный по участнику
    if c1["message_lr"]["enabled"]:
        C = c1["message_lr"]["C"]
        oof = np.zeros((len(tr), 4))
        for a, b in GroupKFold(n_inner, shuffle=True, random_state=seed + off["message_lr"]).split(tr, tr.y, tr.dialog_id):
            oof[b] = _message_lr(ctx["messages"], tr.iloc[a], tr.iloc[b], C, max_iter)
        pte = _message_lr(ctx["messages"], tr, te, C, max_iter)
        for i, n in enumerate(MSG_AGG):
            tr["xm_" + n], te["xm_" + n] = oof[:, i], pte[:, i]
        for d in (tr, te):
            for c in ("xm_mean", "xm_last"):
                d["pt_" + c] = d.groupby("dialog_id")[c].transform("sum") - d[c]

    # стиль
    return join_by_participant(ctx["style"], tr), join_by_participant(ctx["style"], te)


MSG_AGG = ("mean", "max", "min", "last")


def _message_lr(mt, a, b, C, max_iter):
    """Обучение на сообщениях участников a (метка участника -> каждому его сообщению), агрегаты для b."""
    lab = a.set_index(["dialog_id", "participant_index"]).y
    ia = np.where(mt.key.isin(lab.index))[0]
    ib = np.where(mt.key.isin(pd.MultiIndex.from_frame(b[["dialog_id", "participant_index"]])))[0]
    m = LogisticRegression(C=C, max_iter=max_iter).fit(mt.X[ia], lab.loc[mt.key[ia]].values)
    pb = mt.M.iloc[ib][["dialog_id", "participant_index", "last_msg"]].assign(p=m.predict_proba(mt.X[ib])[:, 1])
    g = pb.groupby(["dialog_id", "participant_index"]).p
    agg = pd.DataFrame({"mean": g.mean(), "max": g.max(), "min": g.min(),
                        "last": pb[pb.last_msg == 1].groupby(["dialog_id", "participant_index"]).p.first()})
    return agg.loc[list(zip(b.dialog_id, b.participant_index))].values


def feature_columns(df):
    return [c for c in df.columns if c not in EXCLUDE and not c.startswith("text_")]


def level2_fit_predict(tr, te, cfg):
    """LightGBM, усреднённый по n_seeds. Возвращает предсказания и важности признаков."""
    c2 = cfg["level2"]
    cols = feature_columns(tr)
    pred, imp = np.zeros(len(te)), np.zeros(len(cols))
    for s in range(c2["n_seeds"]):
        params = dict(c2["params"], seed=s, verbose=-1, num_threads=cfg["resources"]["threads"])
        m = lgb.train(params, lgb.Dataset(tr[cols], tr.y), c2["num_boost_round"])
        pred += m.predict(te[cols]) / c2["n_seeds"]
        imp += m.feature_importance("gain") / c2["n_seeds"]
    return pred, pd.Series(imp, index=cols).sort_values(ascending=False)


def _to_dialog(d, cols):
    a = d[d.participant_index == 0].set_index("dialog_id")[cols].add_prefix("a_")
    b = d[d.participant_index == 1].set_index("dialog_id")[cols].add_prefix("b_")
    D = a.join(b)
    y = d.pivot(index="dialog_id", columns="participant_index", values="y").loc[D.index]
    return D, (y[0] * 1 + y[1] * 2).values


def dialog_fit_predict(tr, te, cfg):
    """3-классовый LightGBM на уровне диалога {нет бота, бот 0, бот 1} по признакам обоих участников."""
    c = cfg["level2"]["dialog_model"]
    own = [x for x in feature_columns(tr) if not x.startswith("pt_")]
    Dt, yt = _to_dialog(tr, own)
    Dv, _ = _to_dialog(te.assign(y=0), own)
    params = dict(cfg["level2"]["params"], **c["params"], objective="multiclass", num_class=3,
                  verbose=-1, num_threads=cfg["resources"]["threads"])
    pr = 0
    for s in range(cfg["level2"]["n_seeds"]):
        pr = pr + lgb.train(dict(params, seed=s), lgb.Dataset(Dt, yt), c["num_boost_round"]).predict(Dv)
    pr = pr / cfg["level2"]["n_seeds"]
    m = dict(zip(zip(Dv.index, [0] * len(Dv)), pr[:, 1])) | dict(zip(zip(Dv.index, [1] * len(Dv)), pr[:, 2]))
    return np.array([m[k] for k in zip(te.dialog_id, te.participant_index)])


def predict_level2(tr, te, cfg):
    """Модель участника, при включённой диалоговой модели — взвешенный бленд с ней."""
    p, imp = level2_fit_predict(tr, te, cfg)
    c = cfg["level2"]["dialog_model"]
    if c["enabled"]:
        p = (1 - c["weight"]) * p + c["weight"] * dialog_fit_predict(tr, te, cfg)
    return p, imp


def apply_temperature(p, T):
    """Смягчение уверенности: logit / T."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-np.log(p / (1 - p)) / T))


def run_stack(tr, te, ctx, cfg):
    """Полный стек (1-й + 2-й уровень), усреднённый по сидам внутренних разбиений, затем температура."""
    seeds = cfg["bagging"]["inner_seeds"]
    pred, imp = 0, None
    for s in seeds:
        c = copy.deepcopy(cfg)
        c["seed"] = s
        a, b = level1(tr, te, ctx, c)
        p, i = predict_level2(a, b, c)
        pred = pred + p / len(seeds)
        imp = i if imp is None else imp.add(i, fill_value=0)
    T = cfg["calibration"]["temperature"]
    return apply_temperature(pred, T) if T != 1.0 else pred, (imp / len(seeds)).sort_values(ascending=False)
