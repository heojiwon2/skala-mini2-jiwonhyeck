"""프로젝트 공통 설정: 경로, 시드, 실험 상수"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]           # day2/


def _find_archive():
    for p in [ROOT, *ROOT.parents]:
        if (p / "archive").is_dir():
            return p / "archive"
    raise FileNotFoundError("archive/ 폴더(.mat 원본)를 찾을 수 없습니다. data/README.md 참고")


ARCHIVE = _find_archive()
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

BATCH_FILES = {
    "B1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",   # Train
    "B2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",   # Test (필수)
    "B3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",   # Test (추가)
}

SEED = 42
EARLY_CYCLE = 100          # 예측 시점: 이 사이클까지의 정보만 사용
NOMINAL_Q = 1.1            # 정격 용량 (Ah)
EOL_Q = 0.88               # 수명 종료 기준 = 정격의 80%
CENSOR_MARGIN = 0.01       # 마지막 용량이 EOL + 0.01 보다 크면 EOL 미도달(중도절단)
QDLIN_CYCLES = (4, 5, 10, 100)
PAPER_MAPE = 9.1           # Severson et al. (2019) 목표 성능
