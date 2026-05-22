#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DSS E-commerce Review Project (course-work runnable edition)

Implements:
1) Data cleaning pipeline (UTF-8, schema checks, dedup, conflict mark, text clean)
2) Warehouse-like layered outputs (ODS/DWD/DWS) with bridge table
3) SQL DDL export for MySQL/PostgreSQL style schemas
4) Model 1: score prediction (1-5) using meta + supervised TF-IDF (+ optional BERT embedding)
5) Model 2: weak-supervised multi-label risk classifier
6) Model 3: product-day anomaly detection via IsolationForest
7) DSS decision outputs (key users/products/categories, risk alerts)

Usage:
    python dss.py
or:
    python dss.py --data-dir "商品评论情感预测" --out-dir "outputs"
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer


RANDOM_STATE = 42


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_with_fallback(path: Path, encodings: Sequence[str] = ("utf-8", "gb18030")) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
    if last_err is not None:
        raise last_err
    return pd.read_csv(path)


def try_import_jieba():
    try:
        import jieba  # type: ignore

        return jieba
    except Exception:
        return None


def try_import_bert():
    try:
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        return torch, AutoTokenizer, AutoModel
    except Exception:
        return None


def normalize_user_id(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s


def unix_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, unit="s", errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai")


def clean_text_basic(text: object) -> str:
    if pd.isna(text):
        return ""
    t = str(text)
    t = html.unescape(t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"[\r\n\t]+", " ", t)
    t = re.sub(r"[^\w\u4e00-\u9fff\s]", " ", t)
    # compress elongated repeated chars, keep max 2
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_stopwords() -> set:
    # Keep negation and degree words intentionally
    base = {
        "的",
        "了",
        "是",
        "在",
        "和",
        "与",
        "及",
        "就",
        "都",
        "而",
        "及其",
        "并",
        "一个",
        "这个",
        "那个",
        "我们",
        "你们",
        "他们",
    }
    protected = {"不", "没", "没有", "很", "非常", "太", "挺"}
    return base - protected


def tokenize_cn(text: str, jieba_module, stopwords: set) -> List[str]:
    if not text:
        return []
    if jieba_module is None:
        # fallback: character granularity
        tokens = [ch for ch in text if re.match(r"[\u4e00-\u9fffA-Za-z0-9_]", ch)]
    else:
        tokens = [w.strip() for w in jieba_module.cut(text) if w.strip()]
    return [w for w in tokens if w not in stopwords]


def derive_time_features(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    dt_series = unix_to_datetime(df[ts_col])
    out = df.copy()
    out["review_time"] = dt_series
    out["year"] = dt_series.dt.year
    out["month"] = dt_series.dt.month
    out["day"] = dt_series.dt.day
    out["hour"] = dt_series.dt.hour
    out["weekday"] = dt_series.dt.weekday
    out["is_weekend"] = out["weekday"].isin([5, 6]).astype(int)
    return out


def normalize_score_int(series: pd.Series) -> pd.Series:
    val = pd.to_numeric(series, errors="coerce")
    return val.round().astype("Int64")


def review_conflict_marker(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["user_id", "product_id", "评论时间戳"]
    txt_cols = ["评论标题", "评论内容"]
    g = df.groupby(key_cols, dropna=False)
    conflict_index = g[txt_cols].transform(lambda c: c.nunique(dropna=False)) > 1
    # if title or content has variance in group, mark conflict
    df["is_conflict"] = ((conflict_index["评论标题"]) | (conflict_index["评论内容"])).astype(int)
    # keep latest by review_id lexical order (stable deterministic)
    df = df.sort_values(["user_id", "product_id", "评论时间戳", "review_id"])
    df = df.drop_duplicates(subset=key_cols, keep="last")
    return df


def explode_product_category(product_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Tuple[str, str, int]] = []
    for _, r in product_df[["商品ID", "所属类别"]].iterrows():
        pid = r["商品ID"]
        c = "" if pd.isna(r["所属类别"]) else str(r["所属类别"])
        cats = [x.strip() for x in c.split(",") if x.strip()]
        for idx, cid in enumerate(cats, start=1):
            rows.append((pid, cid, idx))
    return pd.DataFrame(rows, columns=["product_id", "category_id", "category_order"])


def weak_label_risk(df: pd.DataFrame) -> pd.DataFrame:
    quality_kw = ["质量", "掉色", "破损", "坏", "漏", "异味", "假货", "变质", "过敏"]
    logistics_kw = ["物流", "快递", "配送", "发货", "到货", "包裹"]
    service_kw = ["客服", "售后", "服务", "态度", "退款", "维权"]
    desc_kw = ["描述", "不符", "不一致", "实物", "图片", "介绍", "货不对板"]
    text = (df["评论标题"].fillna("") + " " + df["评论内容"].fillna("")).astype(str)
    out = df.copy()
    out["risk_quality"] = text.apply(lambda t: int(any(k in t for k in quality_kw)))
    out["risk_logistics"] = text.apply(lambda t: int(any(k in t for k in logistics_kw)))
    out["risk_service"] = text.apply(lambda t: int(any(k in t for k in service_kw)))
    out["risk_desc_mismatch"] = text.apply(lambda t: int(any(k in t for k in desc_kw)))
    return out


class SupervisedTFIDF:
    """Label-aware TF-IDF features inspired by supervised TF-IDF in the reference paper."""

    def __init__(self, max_features: int = 30000, ngram_range: Tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.class_vocab_weight: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, texts: Sequence[str], y: Sequence[int]) -> "SupervisedTFIDF":
        x = self.vectorizer.fit_transform(texts)  # (n, v)
        lb = LabelBinarizer()
        yb = lb.fit_transform(y)
        if yb.ndim == 1:
            yb = np.vstack([1 - yb, yb]).T
        self.classes_ = lb.classes_
        # class-word average tfidf
        class_sum = yb.T @ x  # (c, v)
        class_count = yb.sum(axis=0).reshape(-1, 1) + 1e-9
        self.class_vocab_weight = class_sum / class_count  # (c, v)
        return self

    def transform(self, texts: Sequence[str]) -> sparse.csr_matrix:
        if self.class_vocab_weight is None:
            raise RuntimeError("SupervisedTFIDF is not fitted")
        x = self.vectorizer.transform(texts)  # (n, v)
        # project to class-aware dimensions
        proj = x @ self.class_vocab_weight.T  # (n, c)
        return sparse.csr_matrix(proj)


class OptionalBertEmbedder:
    def __init__(self, model_name: str = "bert-base-chinese", max_len: int = 100):
        self.model_name = model_name
        self.max_len = max_len
        self.available = False
        self.device = "cpu"
        self.tokenizer = None
        self.model = None
        self.backend = "none"
        self._init_backend()

    def _init_backend(self) -> None:
        dep = try_import_bert()
        if dep is None:
            self.available = False
            self.backend = "none"
            return
        torch, AutoTokenizer, AutoModel = dep
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.available = True
            self.backend = "transformers"
        except Exception:
            self.available = False
            self.backend = "none"

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> sparse.csr_matrix:
        if not self.available:
            return sparse.csr_matrix((len(texts), 0))
        dep = try_import_bert()
        if dep is None:
            return sparse.csr_matrix((len(texts), 0))
        torch = dep[0]
        vecs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = list(texts[i : i + batch_size])
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_len,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                cls_vec = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
                vecs.append(cls_vec)
        arr = np.vstack(vecs) if vecs else np.empty((len(texts), 0), dtype=np.float32)
        return sparse.csr_matrix(arr)


@dataclass
class PipelinePaths:
    data_dir: Path
    out_dir: Path
    ods_dir: Path
    dwd_dir: Path
    dws_dir: Path
    dm_dir: Path
    model_dir: Path
    report_dir: Path
    sql_dir: Path


def make_paths(data_dir: Path, out_dir: Path) -> PipelinePaths:
    p = PipelinePaths(
        data_dir=data_dir,
        out_dir=out_dir,
        ods_dir=out_dir / "ods",
        dwd_dir=out_dir / "dwd",
        dws_dir=out_dir / "dws",
        dm_dir=out_dir / "dm",
        model_dir=out_dir / "models",
        report_dir=out_dir / "reports",
        sql_dir=out_dir / "sql",
    )
    for d in [p.out_dir, p.ods_dir, p.dwd_dir, p.dws_dir, p.dm_dir, p.model_dir, p.report_dir, p.sql_dir]:
        safe_mkdir(d)
    return p


def export_sql_ddl(sql_path: Path) -> None:
    ddl = """
-- DSS warehouse DDL (generic MySQL/PostgreSQL style)
CREATE TABLE IF NOT EXISTS dim_user (
  user_id VARCHAR(64) PRIMARY KEY,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dim_product (
  product_id VARCHAR(64) PRIMARY KEY,
  product_name TEXT,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dim_category (
  category_id VARCHAR(64) PRIMARY KEY,
  category_name TEXT
);

CREATE TABLE IF NOT EXISTS dim_date (
  date_key DATE PRIMARY KEY,
  year INT,
  month INT,
  day INT,
  weekday INT,
  is_weekend INT
);

CREATE TABLE IF NOT EXISTS bridge_product_category (
  product_id VARCHAR(64),
  category_id VARCHAR(64),
  category_order INT,
  PRIMARY KEY(product_id, category_id, category_order)
);

CREATE TABLE IF NOT EXISTS fact_review_train (
  review_id VARCHAR(64) PRIMARY KEY,
  user_id VARCHAR(64),
  product_id VARCHAR(64),
  review_time TIMESTAMP,
  title TEXT,
  content TEXT,
  clean_text TEXT,
  score INT,
  is_conflict INT,
  etl_batch_id VARCHAR(64),
  created_at TIMESTAMP,
  updated_at TIMESTAMP,
  is_deleted INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS fact_review_test (
  review_id VARCHAR(64) PRIMARY KEY,
  user_id VARCHAR(64),
  product_id VARCHAR(64),
  review_time TIMESTAMP,
  title TEXT,
  content TEXT,
  clean_text TEXT,
  is_conflict INT,
  etl_batch_id VARCHAR(64),
  created_at TIMESTAMP,
  updated_at TIMESTAMP,
  is_deleted INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS fact_prediction (
  review_id VARCHAR(64) PRIMARY KEY,
  pred_score INT,
  pred_sentiment VARCHAR(16),
  risk_level VARCHAR(16),
  risk_tags TEXT,
  model_version VARCHAR(64),
  inference_time TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agg_product_daily (
  product_id VARCHAR(64),
  dt DATE,
  review_cnt INT,
  avg_score FLOAT,
  low_score_rate FLOAT,
  PRIMARY KEY(product_id, dt)
);

CREATE TABLE IF NOT EXISTS agg_category_daily (
  category_id VARCHAR(64),
  dt DATE,
  review_cnt INT,
  avg_score FLOAT,
  low_score_rate FLOAT,
  PRIMARY KEY(category_id, dt)
);

CREATE TABLE IF NOT EXISTS agg_user_profile (
  user_id VARCHAR(64) PRIMARY KEY,
  review_cnt INT,
  avg_score FLOAT,
  low_score_rate FLOAT,
  active_days INT,
  favored_categories TEXT,
  user_tier VARCHAR(32)
);

CREATE TABLE IF NOT EXISTS risk_alert_log (
  alert_id VARCHAR(64) PRIMARY KEY,
  product_id VARCHAR(64),
  dt DATE,
  risk_level VARCHAR(16),
  evidence JSON,
  suggestion TEXT,
  owner VARCHAR(64),
  created_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fact_train_product_date ON fact_review_train(product_id, review_time);
CREATE INDEX IF NOT EXISTS idx_fact_train_user ON fact_review_train(user_id);
CREATE INDEX IF NOT EXISTS idx_agg_category_date ON agg_category_daily(category_id, dt);
"""
    sql_path.write_text(ddl.strip() + "\n", encoding="utf-8")


def build_meta_features(df: pd.DataFrame, product_heat_map: Dict[str, int]) -> sparse.csr_matrix:
    text_len = df["clean_text"].fillna("").str.len().astype(float).values.reshape(-1, 1)
    hour = df["hour"].fillna(0).astype(float).values.reshape(-1, 1)
    weekday = df["weekday"].fillna(0).astype(float).values.reshape(-1, 1)
    heat = df["product_id"].map(product_heat_map).fillna(0).astype(float).values.reshape(-1, 1)
    arr = np.hstack([text_len, hour, weekday, heat])
    return sparse.csr_matrix(arr)


def sentiment_from_score(score: int) -> str:
    if score >= 4:
        return "positive"
    if score == 3:
        return "neutral"
    return "negative"


def run_pipeline(data_dir: Path, out_dir: Path) -> None:
    paths = make_paths(data_dir, out_dir)
    ts_now = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    etl_batch_id = f"batch_{ts_now}"

    # 1) Load ODS
    train = read_csv_with_fallback(paths.data_dir / "训练集.csv")
    test = read_csv_with_fallback(paths.data_dir / "测试集.csv")
    product = read_csv_with_fallback(paths.data_dir / "商品信息.csv")
    category = read_csv_with_fallback(paths.data_dir / "商品类别列表.csv")

    train.to_csv(paths.ods_dir / "ods_train_raw.csv", index=False, encoding="utf-8")
    test.to_csv(paths.ods_dir / "ods_test_raw.csv", index=False, encoding="utf-8")
    product.to_csv(paths.ods_dir / "ods_product_raw.csv", index=False, encoding="utf-8")
    category.to_csv(paths.ods_dir / "ods_category_raw.csv", index=False, encoding="utf-8")

    # 2) DWD clean + checks
    def normalize_review_df(df: pd.DataFrame, has_score: bool) -> pd.DataFrame:
        out = df.copy()
        out = out.rename(
            columns={
                "数据ID": "review_id",
                "用户ID": "user_id",
                "商品ID": "product_id",
            }
        )
        out["user_id"] = out["user_id"].apply(normalize_user_id)
        out["评论标题"] = out["评论标题"].fillna("").astype(str)
        out["评论内容"] = out["评论内容"].fillna("").astype(str)
        out = derive_time_features(out, "评论时间戳")
        out["clean_text"] = (out["评论标题"] + " " + out["评论内容"]).apply(clean_text_basic)
        jieba_mod = try_import_jieba()
        stopwords = build_stopwords()
        out["clean_tokens"] = out["clean_text"].apply(lambda x: tokenize_cn(x, jieba_mod, stopwords))
        out["clean_text"] = out["clean_tokens"].apply(lambda x: " ".join(x))
        if has_score:
            out["score"] = normalize_score_int(out["评分"])
            out = out[out["score"].isin([1, 2, 3, 4, 5])]
        out = review_conflict_marker(out)
        out["etl_batch_id"] = etl_batch_id
        out["created_at"] = dt.datetime.now()
        out["updated_at"] = dt.datetime.now()
        out["is_deleted"] = 0
        return out

    train_dwd = normalize_review_df(train, has_score=True)
    test_dwd = normalize_review_df(test, has_score=False)

    # hard constraints
    if train_dwd["review_id"].duplicated().any():
        raise ValueError("train review_id not unique after cleaning")
    if test_dwd["review_id"].duplicated().any():
        raise ValueError("test review_id not unique after cleaning")
    prod_set = set(product["商品ID"].astype(str))
    missing_train_product = (~train_dwd["product_id"].astype(str).isin(prod_set)).sum()
    missing_test_product = (~test_dwd["product_id"].astype(str).isin(prod_set)).sum()
    if missing_train_product > 0 or missing_test_product > 0:
        raise ValueError(f"product foreign key mismatch: train={missing_train_product}, test={missing_test_product}")

    # save cleaned facts
    train_dwd.to_csv(paths.dwd_dir / "fact_review_train.csv", index=False, encoding="utf-8")
    test_dwd.to_csv(paths.dwd_dir / "fact_review_test.csv", index=False, encoding="utf-8")

    # dimensions
    dim_user = pd.DataFrame({"user_id": pd.concat([train_dwd["user_id"], test_dwd["user_id"]]).dropna().unique()})
    dim_product = product.rename(columns={"商品ID": "product_id", "商品名称": "product_name"})[["product_id", "product_name"]]
    dim_category = category.rename(columns={"类别ID": "category_id", "类别名称": "category_name"})
    dim_date = pd.DataFrame({"date_key": pd.to_datetime(pd.concat([train_dwd["review_time"], test_dwd["review_time"]]).dt.date.unique())})
    dim_date["year"] = dim_date["date_key"].dt.year
    dim_date["month"] = dim_date["date_key"].dt.month
    dim_date["day"] = dim_date["date_key"].dt.day
    dim_date["weekday"] = dim_date["date_key"].dt.weekday
    dim_date["is_weekend"] = dim_date["weekday"].isin([5, 6]).astype(int)

    bridge = explode_product_category(product)

    dim_user.to_csv(paths.dwd_dir / "dim_user.csv", index=False, encoding="utf-8")
    dim_product.to_csv(paths.dwd_dir / "dim_product.csv", index=False, encoding="utf-8")
    dim_category.to_csv(paths.dwd_dir / "dim_category.csv", index=False, encoding="utf-8")
    dim_date.to_csv(paths.dwd_dir / "dim_date.csv", index=False, encoding="utf-8")
    bridge.to_csv(paths.dwd_dir / "bridge_product_category.csv", index=False, encoding="utf-8")

    # 3) Model 1 - score prediction
    # split validation
    tr, va = train_test_split(
        train_dwd[["review_id", "clean_text", "score", "product_id", "hour", "weekday"]],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_dwd["score"],
    )
    sup_tfidf = SupervisedTFIDF(max_features=30000, ngram_range=(1, 2))
    sup_tfidf.fit(tr["clean_text"], tr["score"])

    product_heat = train_dwd["product_id"].value_counts().to_dict()
    xtr_sup = sup_tfidf.transform(tr["clean_text"])
    xva_sup = sup_tfidf.transform(va["clean_text"])
    xts_sup = sup_tfidf.transform(test_dwd["clean_text"])

    xtr_meta = build_meta_features(tr, product_heat)
    xva_meta = build_meta_features(va, product_heat)
    xts_meta = build_meta_features(test_dwd, product_heat)

    bert = OptionalBertEmbedder(model_name="bert-base-chinese", max_len=100)
    xtr_bert = bert.encode(tr["clean_text"].tolist())
    xva_bert = bert.encode(va["clean_text"].tolist())
    xts_bert = bert.encode(test_dwd["clean_text"].tolist())

    xtr = sparse.hstack([xtr_sup, xtr_meta, xtr_bert], format="csr")
    xva = sparse.hstack([xva_sup, xva_meta, xva_bert], format="csr")
    xts = sparse.hstack([xts_sup, xts_meta, xts_bert], format="csr")

    clf_score = LogisticRegression(
        multi_class="multinomial",
        solver="saga",
        max_iter=500,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    clf_score.fit(xtr, tr["score"].astype(int))
    pred_va = clf_score.predict(xva)
    pred_ts = clf_score.predict(xts)

    score_metrics = {
        "accuracy": float(accuracy_score(va["score"], pred_va)),
        "macro_f1": float(f1_score(va["score"], pred_va, average="macro")),
        "qwk": float(cohen_kappa_score(va["score"], pred_va, weights="quadratic")),
        "bert_backend": bert.backend,
    }

    submission = pd.DataFrame({"数据ID": test_dwd["review_id"], "评分": pred_ts.astype(int)})
    submission.to_csv(paths.out_dir / "submission.csv", index=False, encoding="utf-8")

    # 4) Model 2 - weak supervised multi-label risk classification
    train_risk = weak_label_risk(train_dwd)
    risk_cols = ["risk_quality", "risk_logistics", "risk_service", "risk_desc_mismatch"]
    y_multi = train_risk[risk_cols].values.astype(int)
    # keep risk modeling on same features, compact (without BERT to save runtime)
    x_all = sparse.hstack(
        [
            sup_tfidf.transform(train_dwd["clean_text"]),
            build_meta_features(train_dwd[["clean_text", "product_id", "hour", "weekday"]], product_heat),
        ],
        format="csr",
    )
    x_train_r, x_valid_r, y_train_r, y_valid_r = train_test_split(
        x_all, y_multi, test_size=0.2, random_state=RANDOM_STATE
    )
    clf_risk = OneVsRestClassifier(
        LogisticRegression(solver="liblinear", max_iter=300, random_state=RANDOM_STATE)
    )
    clf_risk.fit(x_train_r, y_train_r)
    pred_proba_r = clf_risk.predict_proba(x_valid_r)
    threshold = 0.35
    pred_r = (pred_proba_r >= threshold).astype(int)
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        y_valid_r, pred_r, average="micro", zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_valid_r, pred_r, average="macro", zero_division=0
    )
    risk_metrics = {
        "micro_f1": float(f_micro),
        "macro_f1": float(f_macro),
        "micro_precision": float(p_micro),
        "micro_recall": float(r_micro),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "hamming_loss": float(hamming_loss(y_valid_r, pred_r)),
        "threshold": threshold,
    }

    # infer risk tags for test
    x_test_risk = sparse.hstack(
        [
            sup_tfidf.transform(test_dwd["clean_text"]),
            build_meta_features(test_dwd[["clean_text", "product_id", "hour", "weekday"]], product_heat),
        ],
        format="csr",
    )
    test_risk_prob = clf_risk.predict_proba(x_test_risk)
    test_risk_pred = (test_risk_prob >= threshold).astype(int)
    risk_tag_names = {
        0: "质量",
        1: "物流",
        2: "售后",
        3: "描述不符",
    }
    risk_tags = []
    risk_level = []
    for row in test_risk_pred:
        tags = [risk_tag_names[i] for i, v in enumerate(row) if v == 1]
        risk_tags.append(",".join(tags))
        if len(tags) >= 2:
            risk_level.append("high")
        elif len(tags) == 1:
            risk_level.append("medium")
        else:
            risk_level.append("low")

    # 5) Model 3 - anomaly on product-day indicators
    agg_prod = train_dwd.copy()
    agg_prod["date"] = pd.to_datetime(agg_prod["review_time"]).dt.date
    g = agg_prod.groupby(["product_id", "date"], as_index=False).agg(
        review_cnt=("review_id", "count"),
        avg_score=("score", "mean"),
        low_score_rate=("score", lambda s: float((s <= 2).mean())),
    )
    if len(g) > 20:
        iso = IsolationForest(
            n_estimators=200,
            contamination=0.03,
            random_state=RANDOM_STATE,
        )
        an_x = g[["review_cnt", "avg_score", "low_score_rate"]].values
        an_label = iso.fit_predict(an_x)  # -1 anomaly
        an_score = -iso.score_samples(an_x)
        g["is_anomaly"] = (an_label == -1).astype(int)
        g["anomaly_score"] = an_score
    else:
        g["is_anomaly"] = 0
        g["anomaly_score"] = 0.0

    # pseudo auc/f1 against top percentile low-quality proxy label
    proxy_true = ((g["low_score_rate"] > g["low_score_rate"].quantile(0.95)) | (g["avg_score"] < 2.5)).astype(int)
    if proxy_true.nunique() > 1 and g["anomaly_score"].nunique() > 1:
        anomaly_auc = float(roc_auc_score(proxy_true, g["anomaly_score"]))
        anomaly_f1 = float(f1_score(proxy_true, g["is_anomaly"]))
    else:
        anomaly_auc = float("nan")
        anomaly_f1 = float("nan")
    anomaly_metrics = {"proxy_auc": anomaly_auc, "proxy_f1": anomaly_f1}

    # 6) DWS/DM aggregates and decision outputs
    agg_product_daily = g.rename(columns={"date": "dt"})
    agg_product_daily.to_csv(paths.dws_dir / "agg_product_daily.csv", index=False, encoding="utf-8")

    # category daily
    b = bridge.merge(train_dwd[["product_id", "review_time", "score", "review_id"]], on="product_id", how="inner")
    b["dt"] = pd.to_datetime(b["review_time"]).dt.date
    agg_category_daily = (
        b.groupby(["category_id", "dt"], as_index=False)
        .agg(
            review_cnt=("review_id", "count"),
            avg_score=("score", "mean"),
            low_score_rate=("score", lambda s: float((s <= 2).mean())),
        )
        .sort_values(["category_id", "dt"])
    )
    agg_category_daily.to_csv(paths.dws_dir / "agg_category_daily.csv", index=False, encoding="utf-8")

    # user profile
    u = train_dwd.copy()
    u["dt"] = pd.to_datetime(u["review_time"]).dt.date
    user_agg = (
        u.groupby("user_id", as_index=False)
        .agg(
            review_cnt=("review_id", "count"),
            avg_score=("score", "mean"),
            low_score_rate=("score", lambda s: float((s <= 2).mean())),
            active_days=("dt", "nunique"),
        )
        .sort_values("review_cnt", ascending=False)
    )
    user_agg["user_tier"] = np.where(
        (user_agg["review_cnt"] >= user_agg["review_cnt"].quantile(0.9)) & (user_agg["avg_score"] >= 4.0),
        "high_value",
        np.where(user_agg["low_score_rate"] > 0.5, "high_risk", "normal"),
    )
    user_agg.to_csv(paths.dws_dir / "agg_user_profile.csv", index=False, encoding="utf-8")

    # risk alert log
    risk_rows = agg_product_daily[agg_product_daily["is_anomaly"] == 1].copy()
    risk_rows = risk_rows.sort_values("anomaly_score", ascending=False).head(200)
    risk_rows["risk_level"] = np.where(risk_rows["anomaly_score"] > risk_rows["anomaly_score"].quantile(0.8), "high", "medium")
    risk_rows["suggestion"] = np.where(
        risk_rows["risk_level"] == "high",
        "建议立即排查商品质量与履约链路，必要时暂停推广并启动快速退款策略",
        "建议监控近3日评分与差评词，安排客服主动回访",
    )
    risk_rows["owner"] = "售后风控主管"
    risk_rows["alert_id"] = [f"ALERT_{i:06d}" for i in range(1, len(risk_rows) + 1)]
    risk_rows["created_at"] = dt.datetime.now()
    risk_rows["evidence"] = risk_rows.apply(
        lambda r: json.dumps(
            {
                "review_cnt": int(r["review_cnt"]),
                "avg_score": float(r["avg_score"]),
                "low_score_rate": float(r["low_score_rate"]),
                "anomaly_score": float(r["anomaly_score"]),
            },
            ensure_ascii=False,
        ),
        axis=1,
    )
    risk_alert_log = risk_rows[
        ["alert_id", "product_id", "dt", "risk_level", "evidence", "suggestion", "owner", "created_at"]
    ]
    risk_alert_log.to_csv(paths.dm_dir / "risk_alert_log.csv", index=False, encoding="utf-8")

    # decision output: top users/products/categories
    product_rank = agg_product_daily.groupby("product_id", as_index=False).agg(
        review_cnt=("review_cnt", "sum"),
        avg_score=("avg_score", "mean"),
        low_score_rate=("low_score_rate", "mean"),
    )
    # recommendation index
    product_rank["recommend_index"] = (
        0.5 * product_rank["avg_score"] + 0.3 * np.log1p(product_rank["review_cnt"]) - 0.8 * product_rank["low_score_rate"]
    )
    top_products = product_rank.sort_values("recommend_index", ascending=False).head(200)
    risk_products = product_rank.sort_values(["low_score_rate", "avg_score"], ascending=[False, True]).head(200)

    cat_rank = agg_category_daily.groupby("category_id", as_index=False).agg(
        review_cnt=("review_cnt", "sum"),
        avg_score=("avg_score", "mean"),
        low_score_rate=("low_score_rate", "mean"),
    )
    cat_rank["operate_index"] = 0.6 * cat_rank["avg_score"] + 0.2 * np.log1p(cat_rank["review_cnt"]) - 0.7 * cat_rank["low_score_rate"]
    top_categories = cat_rank.sort_values("operate_index", ascending=False).head(100)

    key_users = user_agg.sort_values(["user_tier", "review_cnt", "avg_score"], ascending=[True, False, False]).head(500)

    top_products.to_csv(paths.dm_dir / "key_products.csv", index=False, encoding="utf-8")
    risk_products.to_csv(paths.dm_dir / "risk_products.csv", index=False, encoding="utf-8")
    top_categories.to_csv(paths.dm_dir / "key_categories.csv", index=False, encoding="utf-8")
    key_users.to_csv(paths.dm_dir / "key_users.csv", index=False, encoding="utf-8")

    # prediction fact
    pred_fact = pd.DataFrame(
        {
            "review_id": test_dwd["review_id"],
            "pred_score": pred_ts.astype(int),
            "pred_sentiment": [sentiment_from_score(int(s)) for s in pred_ts],
            "risk_level": risk_level,
            "risk_tags": risk_tags,
            "model_version": "v1_bert_sup_tfidf_iso",
            "inference_time": dt.datetime.now(),
        }
    )
    pred_fact.to_csv(paths.dm_dir / "fact_prediction.csv", index=False, encoding="utf-8")

    export_sql_ddl(paths.sql_dir / "warehouse_ddl.sql")

    # 7) Reports
    raw_score_dist = train_dwd["score"].value_counts(normalize=True).sort_index().to_dict()
    score_dist = {str(int(k)): float(v) for k, v in raw_score_dist.items()}
    data_quality = {
        "rows_train_raw": int(len(train)),
        "rows_train_clean": int(len(train_dwd)),
        "rows_test_raw": int(len(test)),
        "rows_test_clean": int(len(test_dwd)),
        "train_review_id_unique": bool(not train_dwd["review_id"].duplicated().any()),
        "test_review_id_unique": bool(not test_dwd["review_id"].duplicated().any()),
        "missing_product_fk_train": int(missing_train_product),
        "missing_product_fk_test": int(missing_test_product),
        "score_distribution": score_dist,
        "conflict_ratio_train": float(train_dwd["is_conflict"].mean()),
        "conflict_ratio_test": float(test_dwd["is_conflict"].mean()),
        "empty_clean_text_ratio_train": float((train_dwd["clean_text"].str.len() == 0).mean()),
        "empty_clean_text_ratio_test": float((test_dwd["clean_text"].str.len() == 0).mean()),
        "etl_batch_id": etl_batch_id,
    }
    report = {
        "data_quality": data_quality,
        "score_model_metrics": score_metrics,
        "risk_model_metrics": risk_metrics,
        "anomaly_model_metrics": anomaly_metrics,
    }
    (paths.report_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Pipeline completed.")
    print(f"Output directory: {paths.out_dir.resolve()}")
    print(f"Score model metrics: {json.dumps(score_metrics, ensure_ascii=False)}")
    print(f"Risk model metrics: {json.dumps(risk_metrics, ensure_ascii=False)}")
    print(f"Anomaly metrics: {json.dumps(anomaly_metrics, ensure_ascii=False)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DSS e-commerce review pipeline")
    parser.add_argument("--data-dir", default="商品评论情感预测", help="Input dataset directory")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(Path(args.data_dir), Path(args.out_dir))
