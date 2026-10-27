"""Загрузка данных и таблица «один участник диалога — одна строка» с базовыми признаками."""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SEP = " ¦ "


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


@dataclass
class RawData:
    train: dict          # dialog_id -> list of messages (порядок ключей хронологический)
    test: dict
    ytrain: pd.DataFrame
    ytest: pd.DataFrame

    @property
    def dialogs(self) -> dict:
        return {**self.train, **self.test}


def load_raw(data_dir) -> RawData:
    d = Path(data_dir)
    with open(d / "train.json") as f:
        tr = json.load(f)
    with open(d / "test.json") as f:
        te = json.load(f)
    return RawData(tr, te, pd.read_csv(d / "ytrain.csv"), pd.read_csv(d / "ytest.csv"))


def participant_rows(dialogs: dict, ids) -> pd.DataFrame:
    """Признаки участника: структура ходов, эхо, длины, пунктуация + склеенный текст его реплик."""
    out = []
    for d, p in ids:
        ms = dialogs[d]
        mine = [m["text"] for m in ms if m["participant_index"] == p]
        other = [m["text"] for m in ms if m["participant_index"] != p]
        echo, contains, prev_echo, consec = 0, 0, 0, 0
        for i, m in enumerate(ms):
            if m["participant_index"] != p:
                continue
            if i > 0 and ms[i - 1]["participant_index"] == p:
                consec += 1
            if i > 0 and ms[i - 1]["participant_index"] != p:
                prev = ms[i - 1]["text"]
                if norm(prev) == norm(m["text"]):
                    echo += 1
                elif norm(prev) and norm(prev) in norm(m["text"]):
                    contains += 1
        seen_other = set()
        for m in ms:
            if m["participant_index"] != p:
                seen_other.add(norm(m["text"]))
            elif norm(m["text"]) in seen_other:
                prev_echo += 1
        L = [len(t) for t in mine] or [0]
        letters = re.findall(r"[a-zA-Zа-яА-ЯёЁ]", " ".join(mine))
        n = max(len(mine), 1)
        out.append(dict(
            dialog_id=d, participant_index=p,
            n_mine=len(mine), n_other=len(other), n_total=len(ms),
            is_first=int(ms[0]["participant_index"] == p),
            is_last=int(ms[-1]["participant_index"] == p),
            consec=consec,
            echo_rate=echo / n, contains_rate=contains / n, prev_echo_rate=prev_echo / n,
            len_mean=np.mean(L), len_max=max(L), len_min=min(L), len_std=np.std(L),
            other_len_mean=np.mean([len(t) for t in other]) if other else 0,
            uniq_ratio=len({norm(t) for t in mine}) / n,
            lat_share=sum(c.isascii() for c in letters) / max(len(letters), 1),
            upper_share=sum(c.isupper() for c in letters) / max(len(letters), 1),
            q_rate=sum(t.strip().endswith("?") for t in mine) / n,
            excl_rate=sum("!" in t for t in mine) / n,
            punct_end=sum(bool(re.search(r"[.!?]$", t.strip())) for t in mine) / n,
            cap_start=sum(t[:1].isupper() for t in mine) / n,
            trail_space=sum(t != t.rstrip() for t in mine) / n,
            lead_space=sum(t != t.lstrip() for t in mine) / n,
            newline=sum("\n" in t for t in mine) / n,
            digits=sum(bool(re.search(r"\d", t)) for t in mine) / n,
            emoji=sum(bool(re.search(r"[\U0001F300-\U0001FAFF☀-➿]", t)) for t in mine) / n,
            n_words=np.mean([len(t.split()) for t in mine]) if mine else 0,
            empty=sum(not t.strip() for t in mine) / n,
            text_mine=SEP.join(mine),
        ))
    return pd.DataFrame(out)


def build_tables(raw: RawData):
    """train: строки участников с y; test: с ID. Строки идут в порядке json (он хронологический)."""
    ytr, yte = raw.ytrain, raw.ytest
    dtr = participant_rows(raw.train, zip(ytr.dialog_id, ytr.participant_index))
    dtr["y"] = ytr.is_bot.values
    dte = participant_rows(raw.test, zip(yte.dialog_id, yte.participant_index))
    dte["ID"] = yte.ID.values
    for df, dialogs in ((dtr, raw.train), (dte, raw.test)):
        df["jpos"] = df.dialog_id.map({d: i for i, d in enumerate(dialogs)})
    dtr = dtr.sort_values(["jpos", "participant_index"]).drop(columns="jpos").reset_index(drop=True)
    dte = dte.sort_values(["jpos", "participant_index"]).drop(columns="jpos").reset_index(drop=True)
    return dtr, dte
