"""Day2 실험 실행: 피처셋 × 모델 → Train CV / Valid(Hold-out) / Test(B2, B3) MAPE
사용법: python run_experiments.py [피처셋 이름 ...]   (기본: 전부)
"""
import json, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from src import config as C
from src.train import split_b1, select_features, model_zoo, fit, cv_predict, mape

X = pd.read_parquet(C.PROCESSED / "features.parquet")
C.RESULTS.mkdir(parents=True, exist_ok=True)

train, hold = split_b1(X)
b1_all = pd.concat([train, hold])
sel, sel_table = select_features(train, X)
sel_table.round(3).to_csv(C.RESULTS / "feature_selection_b1train.csv")

FEATURE_SETS = {
    "logvar_only": ["log_var"],                                                   # 논문 Variance model
    "paper_discharge": ["log_abs_min", "log_var", "skew", "kurtosis", "Q2", "Qmax_minus_Q2"],  # 논문 Discharge model
    "day1_selected": ["log_var", "kurtosis", "Q1", "slope_91_100"],               # Day1 전략 (3배치 상관 기준)
    "b1_selected": sel,                                                           # Day2: B1 train만으로 재선택
}
targets = sys.argv[1:] or list(FEATURE_SETS)
print(f"B1 train {len(train)}셀 / hold-out {len(hold)}셀 ({hold['protocol'].nunique()} 프로토콜) | B1 선택 피처: {sel}")

rows, preds = [], []
for fs in targets:
    feats = FEATURE_SETS[fs]
    for name, (est, grid) in model_zoo().items():
        t0 = time.time()
        cv = cv_predict(est, grid, train, feats)
        m_tr = fit(est, grid, train[feats], train["log_life"], train["protocol"])
        p_ho = 10 ** m_tr.predict(hold[feats])
        m_all = fit(est, grid, b1_all[feats], b1_all["log_life"], b1_all["protocol"])   # 최종: B1 전체 재학습
        res = dict(feature_set=fs, model=name, n_feat=len(feats),
                   train_cv=mape(train["cycle_life"], cv), valid=mape(hold["cycle_life"], p_ho))
        for b in ["B2", "B3"]:
            T = X[X["batch"] == b]
            p = 10 ** m_all.predict(T[feats])
            res[f"test_{b}"] = mape(T["cycle_life"], p)
            res[f"bias_{b}"] = 100 * np.mean((p - T["cycle_life"]) / T["cycle_life"])   # (+) 과대 예측
            preds.append(pd.DataFrame({"feature_set": fs, "model": name, "cell_id": T.index,
                                       "batch": b, "y": T["cycle_life"].values, "pred": p}))
        res["sec"] = round(time.time() - t0, 1)
        rows.append(res)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)

out = pd.DataFrame(rows)
f = C.RESULTS / "experiments.csv"
if f.exists():
    old = pd.read_csv(f); out = pd.concat([old[~old.feature_set.isin(targets)], out])
out.to_csv(f, index=False)
pd.concat(preds).to_csv(C.RESULTS / f"test_predictions_{'_'.join(targets)}.csv", index=False)
