"""원본 .mat → 셀 단위 피처 테이블(data/processed/features.parquet)
사용법: python run_features.py   (처음 한 번 .mat 캐시를 만들고, 이후에는 캐시를 읽음)
"""
import sys; sys.path.insert(0, ".")
from src import config as C
from src.preprocess import load_all
from src.features import build_feature_table

summary, cells, qdlin, V = load_all()
X = build_feature_table(summary, cells, qdlin)
X.to_parquet(C.PROCESSED / "features.parquet")
print(f"[features] {X.shape[0]} cells × {X.shape[1]} cols → {C.PROCESSED / 'features.parquet'}")
