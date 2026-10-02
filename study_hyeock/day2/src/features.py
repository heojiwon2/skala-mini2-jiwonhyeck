"""셀 단위 피처 생성. 모든 피처는 cycle 2 ~ 100 (또는 실험 전 정보)만 사용한다."""
import re
import numpy as np
import pandas as pd
from scipy import stats

from . import config as C

POLICY_RE = re.compile(r"([\d.]+)C\((\d+)%\)-([\d.]+)C")

DQ_FEATURES = ["log_var", "log_abs_min", "log_abs_mean", "skew", "kurtosis", "dq_at_2V"]
SUMMARY_FEATURES = ["Q2", "Q100", "Qmax_minus_Q2", "slope_2_100", "slope_91_100",
                    "IR2", "IR_min", "IR_diff", "Tavg_mean", "Tmax_max", "Tmin_min", "chargetime_2_6"]
POLICY_FEATURES = ["C1", "Q1", "C2"]
ALL_FEATURES = DQ_FEATURES + SUMMARY_FEATURES + POLICY_FEATURES
LEAKY = ["knee", "knee_frac", "late_rate", "accel"]   # 100사이클 이후 정보 → 사용 금지 (Day1 I15)


def early_summary(summary):
    """① 먼저 cycle 2~100으로 자르고 ② 그 안에서만 스무딩 → 101사이클 이후 정보가 섞이지 않음
    (Day1 노트북은 스무딩 후 자르는 순서라 cycle 96~100이 101~105를 참조하는 문제가 있었음)"""
    d = summary[summary["cycle"].between(2, C.EARLY_CYCLE)].copy()
    d.loc[~d["QDischarge"].between(0.5, 1.3), "QDischarge"] = np.nan       # 물리적으로 불가능한 값
    d["Q_s"] = d.groupby("cell_id")["QDischarge"].transform(
        lambda s: s.rolling(11, center=True, min_periods=1).median())
    d.loc[d["IR"] <= 0, "IR"] = np.nan                                        # IR 미측정(0)
    for t in ["Tavg", "Tmax", "Tmin"]:
        d.loc[~d[t].between(20, 50), t] = np.nan                              # 센서 이상(400°C 등)
    return d


def summary_features(summary):
    d = early_summary(summary)
    rows = []
    for cid, g in d.groupby("cell_id"):
        g = g.dropna(subset=["Q_s"])
        c, q = g["cycle"].to_numpy(float), g["Q_s"].to_numpy()
        last = c > C.EARLY_CYCLE - 10
        ir = g["IR"].dropna()
        rows.append(dict(
            cell_id=cid,
            Q2=q[0], Q100=q[-1], Qmax_minus_Q2=q.max() - q[0],
            slope_2_100=np.polyfit(c, q, 1)[0] * 100,                 # Ah / 100 cycle
            slope_91_100=np.polyfit(c[last], q[last], 1)[0] * 100,
            IR2=ir.iloc[0] if len(ir) else np.nan,
            IR_min=ir.min() if len(ir) else np.nan,
            IR_diff=(ir.iloc[-1] - ir.iloc[0]) if len(ir) else np.nan,
            Tavg_mean=g["Tavg"].mean(), Tmax_max=g["Tmax"].max(), Tmin_min=g["Tmin"].min(),
            chargetime_2_6=g.loc[g["cycle"] <= 6, "chargetime"].mean(),       # Day1 노트북과 같은 평균
        ))
    return pd.DataFrame(rows).set_index("cell_id")


def dq_features(qdlin, a=100, b=10):
    """ΔQ(V) = Q_a(V) - Q_b(V) 곡선의 요약 통계량 (Severson et al. 2019)"""
    rows = {}
    for cid, d in qdlin.items():
        dq = d[a] - d[b]
        rows[cid] = dict(
            log_var=np.log10(np.var(dq)),
            log_abs_min=np.log10(np.abs(dq.min())),
            log_abs_mean=np.log10(np.abs(dq.mean())),
            skew=stats.skew(dq), kurtosis=stats.kurtosis(dq),
            dq_at_2V=dq[-1],
        )
    return pd.DataFrame(rows).T


def policy_features(cells):
    def parse(p):
        m = POLICY_RE.match(p)
        return (float(m[1]), float(m[2]), float(m[3])) if m else (np.nan,) * 3
    return pd.DataFrame(cells["policy"].map(parse).tolist(), index=cells["cell_id"], columns=POLICY_FEATURES)


def build_feature_table(summary, cells, qdlin):
    X = (cells.set_index("cell_id")
              .join(dq_features(qdlin))
              .join(summary_features(summary))
              .join(policy_features(cells)))
    X["log_life"] = np.log10(X["cycle_life"])
    return X
