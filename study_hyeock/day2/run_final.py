import sys; sys.path.insert(0, ".")
import pandas as pd
from src import config as C
from src.report import run_final, performance_table, error_breakdown, recalibration_probe, figures

X = pd.read_parquet(C.PROCESSED / "features.parquet")
P, model, coef = run_final(X)
perf = performance_table(P); perf.to_csv(C.RESULTS / "model_performance.csv", index=False)
P.to_csv(C.RESULTS / "final_predictions.csv")
eb = error_breakdown(P); eb.to_csv(C.RESULTS / "error_breakdown.csv")
probe = recalibration_probe(P)
pd.DataFrame(probe, index=["mean", "std"]).T.round(2).to_csv(C.RESULTS / "recalibration_probe.csv")
figures(P, coef)
print(perf.to_string(index=False)); print(eb); print("coef\n", coef.round(4)); print("recal", probe)
print(P[P.set=="B2"].sort_values("ape",ascending=False).head(5)[["policy","y","pred","ape"]].round(0))
print(P[P.set=="B3"].sort_values("ape",ascending=False).head(5)[["policy","y","pred","ape"]].round(0))
