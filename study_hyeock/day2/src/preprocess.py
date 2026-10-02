"""원본 .mat(MATLAB v7.3 = HDF5) → 정제된 셀/사이클 테이블

- summary : 사이클 단위 요약 (QDischarge, IR, 온도, 충전시간)
- cells   : 셀 단위 메타 (배치, 정책, cycle_life, 중도절단 여부)
- qdlin   : 셀별 Qdlin(전압축 보간 방전용량) - cycle 4, 5, 10, 100
"""
import sys
import h5py
import numpy as np
import pandas as pd

from . import config as C


def _mat_str(f, ref):
    return "".join(chr(c) for c in f[ref][()].ravel())


def load_batch(name):
    """배치 하나를 읽어 (summary, cells, qdlin, V) 반환. h5py로 필요한 부분만 읽는다."""
    rows, meta, qd = [], [], {}
    with h5py.File(C.ARCHIVE / C.BATCH_FILES[name], "r") as f:
        b = f["batch"]
        V = f[b["Vdlin"][0, 0]][()].ravel()
        for i in range(b["summary"].shape[0]):
            cid = f"{name}_{i:02d}"
            s = f[b["summary"][i, 0]]
            df = pd.DataFrame({k: s[k][()].ravel() for k in s.keys()})
            df.insert(0, "cell_id", cid)
            rows.append(df)
            meta.append(dict(cell_id=cid, batch=name,
                             policy=_mat_str(f, b["policy_readable"][i, 0]),
                             cycle_life=f[b["cycle_life"][i, 0]][()].item()))
            cyc = f[b["cycles"][i, 0]]
            # cycles 배열은 0부터, 사이클 번호는 1부터 → cycle k = 인덱스 k-1 (Day1 EDA에서 확인)
            qd[cid] = {k: f[cyc["Qdlin"][k - 1, 0]][()].ravel() for k in C.QDLIN_CYCLES}
    summary = pd.concat(rows, ignore_index=True)
    summary["cycle"] = summary["cycle"].astype(int)
    summary["batch"] = name
    return summary, pd.DataFrame(meta), qd, V


def build_cache(names=None):
    """배치별 캐시 생성 (배치당 수십 초). 이미 있으면 건너뜀."""
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    for name in names or C.BATCH_FILES:
        if (C.PROCESSED / f"{name}_summary.parquet").exists():
            continue
        s, m, qd, V = load_batch(name)
        s.to_parquet(C.PROCESSED / f"{name}_summary.parquet", index=False)
        m.to_parquet(C.PROCESSED / f"{name}_cells.parquet", index=False)
        np.savez_compressed(C.PROCESSED / f"{name}_qdlin.npz", V=V,
                            **{f"{cid}__c{k}": a for cid, d in qd.items() for k, a in d.items()})
        print(f"[cache] {name}: {m.shape[0]} cells, {s.shape[0]} rows")


def flag_censored(cells, summary):
    """EOL(0.88Ah)에 닿기 전에 기록이 끝난 셀 = cycle_life가 실제 수명이 아님 (Day1 I13)"""
    s = summary[(summary["cycle"] > 1) & summary["QDischarge"].between(0.5, 1.3)]
    q_end = s.groupby("cell_id")["QDischarge"].apply(lambda x: x.iloc[-10:].median())
    cells = cells.copy()
    cells["q_end"] = cells["cell_id"].map(q_end)
    cells["censored"] = cells["cycle_life"].notna() & (cells["q_end"] > C.EOL_Q + C.CENSOR_MARGIN)
    return cells


def load_all():
    """캐시를 읽어 세 배치를 합친다. cycle_life 결측 셀(정답 없음)은 제외."""
    build_cache()
    S, M, Q = [], [], {}
    V = None
    for name in C.BATCH_FILES:
        S.append(pd.read_parquet(C.PROCESSED / f"{name}_summary.parquet"))
        M.append(pd.read_parquet(C.PROCESSED / f"{name}_cells.parquet"))
        z = np.load(C.PROCESSED / f"{name}_qdlin.npz")
        V = z["V"]
        for key in z.files:
            if key != "V":
                cid, k = key.split("__c")
                Q.setdefault(cid, {})[int(k)] = z[key]
    summary, cells = pd.concat(S, ignore_index=True), pd.concat(M, ignore_index=True)
    cells = flag_censored(cells, summary)
    n_all = len(cells)
    cells = cells[cells["cycle_life"].notna()].reset_index(drop=True)
    cells["newstructure"] = cells["policy"].str.contains("newstructure").astype(int)
    # 같은 충전 조건 = 같은 그룹 (newstructure 표기 제거) → 누수 방지 분할 단위
    cells["protocol"] = cells["policy"].str.replace("-newstructure", "", regex=False)
    print(f"[load] {n_all} cells → cycle_life 있는 {len(cells)} cells "
          f"(중도절단 {int(cells['censored'].sum())}셀 표시)")
    return summary, cells, Q, V


if __name__ == "__main__":
    build_cache(sys.argv[1:] or None)
