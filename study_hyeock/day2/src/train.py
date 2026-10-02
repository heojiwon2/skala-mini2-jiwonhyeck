"""분할 · 피처 선택 · 모델 학습/평가

데이터 분할 (누수 방지)
  B1(학습) ─┬─ Hold-out : 충전 프로토콜 단위로 20% 통째로 분리  → Valid
            └─ 나머지   : 프로토콜 GroupKFold(5)               → Train CV, 하이퍼파라미터 튜닝
  최종 모델 = B1 전체로 재학습 → B2(Test), B3(추가 Test) 각 1회 평가
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import ALL_FEATURES

# ---------------------------------------------------------------- 분할
def split_b1(X, include_censored=False, seed=C.SEED, test_size=0.2):
    """B1을 프로토콜 단위로 train / hold-out 분할. 중도절단 셀은 기본 제외(Day1 I13)."""
    b1 = X[X["batch"] == "B1"]
    clean = b1[~b1["censored"]]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, ho_idx = next(gss.split(clean, groups=clean["protocol"]))
    train, hold = clean.iloc[tr_idx], clean.iloc[ho_idx]
    if include_censored:          # 중도절단 셀은 학습 쪽에만 추가 (정답이 부정확하므로 평가에는 쓰지 않음)
        train = pd.concat([train, b1[b1["censored"]]])
    return train, hold


# ---------------------------------------------------------------- 피처 선택 (B1 train만 사용)
def select_features(train, X_unlabeled, r_min=0.3, redund=0.7, ks_max=0.75, std_ratio_min=0.25):
    """① 관련성: B1 train에서 |r(log_life)| ≥ r_min
       ② 이식성: 테스트 배치의 '입력값 분포만' 보고(정답 X) 분포가 크게 다르거나 거의 상수인 피처 제외
       ③ 중복:  이미 고른 피처와 |r| ≥ redund 이면 제외
    """
    r = train[ALL_FEATURES].corrwith(train["log_life"])
    rows = []
    for f in ALL_FEATURES:
        rec = dict(feature=f, r_train=r[f])
        for b in ["B2", "B3"]:
            xb = X_unlabeled.loc[X_unlabeled["batch"] == b, f].dropna()
            xt = train[f].dropna()
            rec[f"KS_{b}"] = ks_2samp(xt, xb).statistic
            rec[f"std_ratio_{b}"] = xb.std() / xt.std()
        rows.append(rec)
    T = pd.DataFrame(rows).set_index("feature")
    T["relevant"] = T["r_train"].abs() >= r_min
    T["transfer_ok"] = (T[["KS_B2", "KS_B3"]].max(axis=1) <= ks_max) & \
                       (T[["std_ratio_B2", "std_ratio_B3"]].min(axis=1) >= std_ratio_min)
    chosen = []
    for f in T[T["relevant"] & T["transfer_ok"]].sort_values("r_train", key=abs, ascending=False).index:
        if all(abs(train[f].corr(train[s])) < redund for s in chosen):
            chosen.append(f)
    T["selected"] = T.index.isin(chosen)

    def why(f):
        if f in chosen: return "선택"
        if not T.at[f, "relevant"]: return f"약함 (|r| < {r_min})"
        if not T.at[f, "transfer_ok"]: return "테스트 배치에서 분포 이동/상수"
        return f"중복 (|r| ≥ {redund})"
    T["status"] = [why(f) for f in T.index]
    return chosen, T.sort_values("r_train", key=abs, ascending=False)


# ---------------------------------------------------------------- 모델
def _pipe(est):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", est)])


def model_zoo():
    gpr_kernel = (ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=2.0, length_scale_bounds=(0.3, 30.0))
                  + WhiteKernel(0.1, (1e-3, 0.5)))   # 노이즈 상한을 둬서 "평균만 예측"으로 붕괴하는 것 방지
    return {
        "Dummy (평균)": (_pipe(DummyRegressor()), {}),
        "Linear": (_pipe(LinearRegression()), {}),
        "ElasticNet": (_pipe(ElasticNet(max_iter=50000)),
                       {"model__alpha": [0.001, 0.003, 0.01, 0.03, 0.1], "model__l1_ratio": [0.1, 0.5, 0.9]}),
        "GPR": (_pipe(GaussianProcessRegressor(kernel=gpr_kernel, normalize_y=True,
                                               n_restarts_optimizer=3, random_state=C.SEED)), {}),
        "RandomForest": (_pipe(RandomForestRegressor(n_estimators=200, random_state=C.SEED)),
                         {"model__max_depth": [2, 3], "model__min_samples_leaf": [2, 4]}),
        "GradientBoosting": (_pipe(GradientBoostingRegressor(random_state=C.SEED, subsample=0.8)),
                             {"model__max_depth": [1, 2], "model__n_estimators": [100, 300],
                              "model__learning_rate": [0.05]}),
    }


def fit(est, grid, Xtr, ytr, groups, n_splits=5):
    """그리드가 있으면 프로토콜 GroupKFold 안에서만 튜닝 (Hold-out / 테스트는 튜닝에 쓰지 않음)"""
    if not grid:
        return clone(est).fit(Xtr, ytr)
    gs = GridSearchCV(clone(est), grid, cv=GroupKFold(min(n_splits, groups.nunique())),
                      scoring="neg_mean_absolute_error")
    gs.fit(Xtr, ytr, groups=groups)
    return gs.best_estimator_


def mape(y_true_cycles, y_pred_cycles):
    return 100 * np.mean(np.abs(y_pred_cycles - y_true_cycles) / y_true_cycles)


def cv_predict(est, grid, df, feats, n_splits=5):
    """Train CV: 바깥 GroupKFold 예측 (각 fold 안에서 다시 GroupKFold로 튜닝 = nested CV)"""
    pred = pd.Series(index=df.index, dtype=float)
    for tr, va in GroupKFold(n_splits).split(df, groups=df["protocol"]):
        a, b = df.iloc[tr], df.iloc[va]
        m = fit(est, grid, a[feats], a["log_life"], a["protocol"])
        pred.iloc[va] = 10 ** m.predict(b[feats])
    return pred
