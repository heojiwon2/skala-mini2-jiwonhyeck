"""최종 모델 학습 → 성능 리포트(포맷) · 오류 분석 · 그림"""
import logging, warnings
warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from .train import split_b1, model_zoo, fit, cv_predict, mape

FINAL_FEATURES = ["log_abs_min", "log_var", "skew", "kurtosis", "Q2", "Qmax_minus_Q2"]   # 논문 Discharge model 피처
FINAL_MODEL = "ElasticNet"
COLORS = {"B1 CV": "#2a78d6", "B1 Hold-out": "#184f95", "B2": "#eb6834", "B3": "#1baf7a"}


def run_final(X):
    est, grid = model_zoo()[FINAL_MODEL]
    train, hold = split_b1(X)
    b1 = pd.concat([train, hold])
    cv = cv_predict(est, grid, train, FINAL_FEATURES)
    m_tr = fit(est, grid, train[FINAL_FEATURES], train["log_life"], train["protocol"])
    ho = pd.Series(10 ** m_tr.predict(hold[FINAL_FEATURES]), index=hold.index)
    m = fit(est, grid, b1[FINAL_FEATURES], b1["log_life"], b1["protocol"])
    P = [pd.DataFrame({"set": "B1 CV", "y": train["cycle_life"], "pred": cv}),
         pd.DataFrame({"set": "B1 Hold-out", "y": hold["cycle_life"], "pred": ho})]
    for b in ["B2", "B3"]:
        T = X[X["batch"] == b]
        P.append(pd.DataFrame({"set": b, "y": T["cycle_life"], "pred": 10 ** m.predict(T[FINAL_FEATURES])}))
    P = pd.concat(P)
    P = P.join(X[["policy", "protocol", "newstructure"]])
    P["ape"] = 100 * (P["pred"] - P["y"]).abs() / P["y"]
    P["signed"] = 100 * (P["pred"] - P["y"]) / P["y"]
    coef = pd.Series(m.named_steps["model"].coef_, index=FINAL_FEATURES)
    return P, m, coef


def performance_table(P):
    g = P.groupby("set")["ape"].mean()
    tr, va, b2, b3 = g["B1 CV"], g["B1 Hold-out"], g["B2"], g["B3"]
    rows = [
        ("Train (Batch 1 CV)", tr, "프로토콜 GroupKFold(5), nested 튜닝"),
        ("Valid (Batch 1 Hold-out)", va, "프로토콜 단위 20% hold-out (7셀)"),
        ("Test (Batch 2)", b2, "39셀"),
        ("Gap (Train-Valid)", va - tr, "Valid − Train, (+) : 과적합 의심"),
        ("Gap (Valid-Test)", b2 - va, "Test − Valid, (+) : 배치간 일반화 저하 의심"),
        ("Gap (Target-Test)", b2 - C.PAPER_MAPE, f"Test − Target, Target : 원논문 {C.PAPER_MAPE}%"),
        ("Test (Batch 3)", b3, "44셀 (추가)"),
        ("Gap (Batch2-Batch3)", b2 - b3, "B2 − B3"),
        ("Gap (Target-Test, B3)", b3 - C.PAPER_MAPE, f"B3 − Target {C.PAPER_MAPE}%"),
    ]
    return pd.DataFrame(rows, columns=["구분", "MAPE (%)", "비고"]).round({"MAPE (%)": 2})


def error_breakdown(P):
    t = P[P["set"].isin(["B2", "B3"])].copy()
    t["group"] = np.where(t["set"] == "B3", "B3 (전부 newstructure)",
                          np.where(t["newstructure"] == 1, "B2 newstructure", "B2 기존 구조"))
    return t.groupby("group").agg(n=("y", "size"), life_median=("y", "median"), pred_median=("pred", "median"),
                                  MAPE=("ape", "mean"), bias=("signed", "mean")).round(1)


def recalibration_probe(P, k=5, reps=200, seed=C.SEED):
    """[진단 실험 · 최종 성능 아님] B2 셀 k개의 실제 수명만 알면 오차가 얼마나 줄어드는가?
    log 공간에서 '평균 오프셋'만 보정 → 남은 셀로 MAPE 계산. 오차가 '배치 기준선 이동'인지 확인용."""
    rng = np.random.default_rng(seed)
    out = {}
    for b in ["B2", "B3"]:
        t = P[P["set"] == b]
        res = []
        for _ in range(reps):
            idx = rng.choice(len(t), k, replace=False)
            cal, rest = t.iloc[idx], t.drop(t.index[idx])
            off = np.median(np.log10(cal["y"]) - np.log10(cal["pred"]))
            res.append(mape(rest["y"], rest["pred"] * 10 ** off))
        out[b] = (np.mean(res), np.std(res))
    return out


def figures(P, coef):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": ["AppleGothic", "Noto Sans CJK JP", "DejaVu Sans"], "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    # 1) Parity plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, sets in zip(axes, [["B1 CV", "B1 Hold-out"], ["B2", "B3"]]):
        for s in sets:
            t = P[P["set"] == s]
            ax.scatter(t["y"], t["pred"], s=34, color=COLORS[s], edgecolor="white", lw=.7,
                       label=f"{s}  MAPE {t['ape'].mean():.1f}%")
        lim = [300, 2100]
        ax.plot(lim, lim, color="#52514e", lw=1)
        ax.fill_between(lim, [l * .9 for l in lim], [l * 1.1 for l in lim], color="#c3c2b7", alpha=.25, lw=0, label="±10%")
        ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="실제 수명 (cycle)", ylabel="예측 수명 (cycle)")
        ax.legend(frameon=False, loc="upper left")
    axes[0].set_title("학습 배치 (B1): CV · Hold-out", loc="left", fontweight="bold")
    axes[1].set_title("테스트 배치: B2 · B3", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(C.FIGURES / "01_parity.png", dpi=150); plt.close(fig)
    # 2) 오류 분석: 그룹별 부호 있는 오차
    t = P[P["set"].isin(["B2", "B3"])].copy()
    t["group"] = np.where(t["set"] == "B3", "B3", np.where(t["newstructure"] == 1, "B2 newstructure", "B2 기존 구조"))
    order = ["B2 기존 구조", "B2 newstructure", "B3"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    data = [t.loc[t["group"] == g, "signed"] for g in order]
    axes[0].boxplot(data, tick_labels=order, showfliers=False)
    for i, d in enumerate(data):
        axes[0].scatter(np.full(len(d), i + 1) + np.random.default_rng(0).uniform(-.12, .12, len(d)), d, s=18,
                        color=["#eb6834", "#c46b2c", "#1baf7a"][i], alpha=.8)
    axes[0].axhline(0, color="#52514e", lw=1)
    axes[0].set(ylabel="(예측 − 실제) / 실제  (%)", title="부호 있는 오차: (+) = 수명을 길게 예측")
    b3 = t[t["group"] == "B3"]
    axes[1].scatter(b3["y"], b3["signed"], s=30, color="#1baf7a", edgecolor="white")
    axes[1].axvline(P.loc[P["set"].isin(["B1 CV", "B1 Hold-out"]), "y"].max(), color="#2a78d6", ls="--", lw=1)
    axes[1].text(P.loc[P["set"].isin(["B1 CV", "B1 Hold-out"]), "y"].max() + 20, axes[1].get_ylim()[1] * .9,
                 "B1 학습 최대 수명", color="#2a78d6", fontsize=9)
    axes[1].axhline(0, color="#52514e", lw=1)
    axes[1].set(xlabel="실제 수명 (cycle)", ylabel="부호 있는 오차 (%)", title="B3: 학습 범위 밖 장수명일수록 짧게 예측")
    fig.tight_layout(); fig.savefig(C.FIGURES / "02_error_analysis.png", dpi=150); plt.close(fig)
    # 3) 계수
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    c = coef.sort_values()
    ax.barh(c.index, c.values, color=np.where(c.values > 0, "#2a78d6", "#eb6834"))
    ax.axvline(0, color="#52514e", lw=1)
    ax.set(title="ElasticNet 계수 (표준화 피처, 타깃 = log10 수명)", xlabel="계수")
    fig.tight_layout(); fig.savefig(C.FIGURES / "03_coefficients.png", dpi=150); plt.close(fig)
