import sys; sys.path.insert(0, ".")
import pandas as pd
from src import config as C
from src.report import (run_final, performance_table, metrics_by_set, error_breakdown, recalibration_probe, figures,
                        FINAL_MODEL, BASELINE)

X = pd.read_parquet(C.PROCESSED / "features.parquet")
P, model, coef = run_final(X)
perf = performance_table(P); perf.to_csv(C.RESULTS / "model_performance.csv", index=False)
P.to_csv(C.RESULTS / "final_predictions.csv")
# Day1 전략 3-2(지표: RMSE · MAPE · bias) · 3-3(기준선 = log_var 선형) 기준 보조 표
P_base, _, _ = run_final(X, feats=BASELINE[1], model=BASELINE[0])
mbs = metrics_by_set({f"최종 ({FINAL_MODEL})": P, "기준선 (log_var 선형)": P_base})
mbs.to_csv(C.RESULTS / "metrics_by_set.csv")
eb = error_breakdown(P); eb.to_csv(C.RESULTS / "error_breakdown.csv")
probe = recalibration_probe(P)
pd.DataFrame(probe, index=["mean", "std"]).T.round(2).to_csv(C.RESULTS / "recalibration_probe.csv")
figures(P, coef)
print(perf.to_string(index=False)); print(mbs); print(eb); print("coef\n", coef.round(4)); print("recal", probe)
print(P[P.set=="B2"].sort_values("ape",ascending=False).head(5)[["policy","y","pred","ape"]].round(0))
print(P[P.set=="B3"].sort_values("ape",ascending=False).head(5)[["policy","y","pred","ape"]].round(0))
