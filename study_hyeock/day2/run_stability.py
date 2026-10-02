"""B1 안에서만: Hold-out을 seed 10개로 바꿔가며 후보 모델의 안정성 확인 + 중도절단 포함/제외 비교
(B2/B3는 보지 않음 → 최종 모델 선택에 테스트 정보가 들어가지 않게)"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from src import config as C
from src.train import split_b1, model_zoo, fit, cv_predict, mape

X = pd.read_parquet(C.PROCESSED / "features.parquet")
SETS = {"logvar_only": ["log_var"],
        "paper_discharge": ["log_abs_min", "log_var", "skew", "kurtosis", "Q2", "Qmax_minus_Q2"]}
CANDS = [("logvar_only", "Linear"), ("paper_discharge", "Linear"),
         ("paper_discharge", "ElasticNet"), ("paper_discharge", "GPR")]
zoo = model_zoo(); rows = []
for inc in [False, True]:
    for fs, mn in CANDS:
        est, grid = zoo[mn]; feats = SETS[fs]
        for seed in range(10):
            tr, ho = split_b1(X, include_censored=inc, seed=seed)
            m = fit(est, grid, tr[feats], tr["log_life"], tr["protocol"])
            rows.append(dict(censored="포함" if inc else "제외", feature_set=fs, model=mn, seed=seed,
                             valid=mape(ho["cycle_life"], 10 ** m.predict(ho[feats]))))
R = pd.DataFrame(rows)
S = R.groupby(["censored", "feature_set", "model"])["valid"].agg(["mean", "std", "min", "max"]).round(2)
print(S); S.to_csv(C.RESULTS / "holdout_stability_b1.csv")
