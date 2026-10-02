# ESS 배터리 수명 예측

초기 100사이클 데이터만으로 리튬이온 셀의 전체 수명(Cycle Life)을 예측해, ESS 셀 교체 계획과 입고 셀 선별에 쓸 수 있는지 확인합니다.

## 프로젝트 개요
- 데이터셋 : MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습 데이터 : Batch 1 (2017-05-12)
- 평가 데이터 : Batch 2 (2018-02-20), 추가 평가 Batch 3 (2018-04-12)
- 태스크 : **Regression** (Cycle Life 예측, Target = `log10(cycle_life)`, 평가 지표 MAPE)

## 파일 구조
```
├── study_jiwon/                   # 허지원
│   ├── day_1_jiwon.ipynb          # Day 1: 데이터 로드, EDA 1~5, 데이터 점검, 모델 설계 전략
│   ├── day_2_jiwon.ipynb          # Day 2: 모델 개발 및 평가
│   └── results/
│       ├── model_performance.csv  # 성능 표 (과제 Format)
│       ├── metrics_by_set.csv     # 보조 지표: RMSE · MAPE · bias, 기준선 대비
│       └── predictions.csv        # 셀별 예측값
├── study_hyeock/                  # 심혁
│   ├── day1_hyeock.ipynb          # Day 1: EDA Q1~Q3
│   ├── day1_deliverable_Q1-Q3.md  # Day 1 보고서 (시사점 I1 ~ I15 정의)
│   ├── figures/                   # Day 1 보고서 그림
│   └── day2/                      # Day 2: 같은 모델링을 스크립트로 구현
│       ├── src/                   # config · preprocess · features · train · report
│       ├── run_features.py        # .mat → 셀 단위 피처 테이블
│       ├── run_experiments.py     # 피처셋 × 모델 비교
│       ├── run_stability.py       # Hold-out 분할 seed 10회 안정성
│       ├── run_final.py           # 최종 모델 평가 · 오류 분석 · 그림
│       └── results/
├── archive/                       # 원본 .mat (용량 문제로 git 제외)
├── requirements.txt
└── README.md
```
두 사람이 같은 설계를 함께 진행했고, 허지원의 작업은 `study_jiwon/`, 심혁의 작업은 `study_hyeock/`에 있습니다. Day 2의 노트북과 스크립트는 같은 결과를 냅니다.

## 환경 설정
```bash
git clone https://github.com/heojiwon2/skala-mini2-jiwonhyeck.git
cd skala-mini2-jiwonhyeck
pip install -r requirements.txt
```
- 원본 `.mat` 파일 4개(`*_batchdata_updated_struct_errorcorrect.mat`)를 `archive/`에 둡니다.
- 노트북: `study_jiwon/day_1_jiwon.ipynb` → `study_jiwon/day_2_jiwon.ipynb` 순서로 위에서부터 실행합니다.
- 스크립트: `cd study_hyeock/day2` 후 `run_features.py` → `run_experiments.py` → `run_stability.py` → `run_final.py` 순서로 실행합니다.

## EDA

- **Cycle Life 분포**
	- 중앙값 Batch 1 858 / Batch 2 472 / Batch 3 1,006. 단수명(< 500) 비율 0% / 72% / 0%, 장수명(> 1,000) 비율 22% / 8% / 52%
	- Batch 2의 고수명 이상치 9셀은 전부 `newstructure` 셀로, 같은 충전 정책의 기존 셀보다 수명이 1.6 ~ 2.2배
	- 핵심 발견 : 학습 배치(Batch 1)와 테스트 배치의 수명 범위가 겹치지 않아, 범위 밖 예측(외삽)이 가능한 모델이 필요함

- **열화 곡선 분석**
	- 수명의 약 60%까지는 0.005 Ah / 100 cycle 안팎으로 거의 일정하다가 이후 급격히 가속 (마지막 20% 구간 속도 = 전체 평균의 약 3.4배)
	- Knee point는 세 배치 모두 수명의 약 75 ~ 79% 지점
	- 핵심 발견 : 100사이클 시점의 용량은 수명과 상관이 약함 (r = 0.32 / −0.28 / 0.34) → 용량 값보다 곡선 모양 기반 피처가 필요함

- **ΔQ(V) 곡선 분석**
	- ΔQ(V) = Q100(V) − Q10(V)는 3.2V 근처에서 꺼지고 2.9 ~ 3.0V에서 가장 깊어짐
	- 단수명 셀일수록 골이 깊음 (장수명 대비 1.5 ~ 2.5배)
	- 핵심 발견 : `log_var` = log10 var(ΔQ)가 세 배치 모두 가장 강한 피처 (r = −0.87 / −0.92 / −0.76). 다만 Batch 1으로 맞춘 직선 기준 Batch 2는 수명이 약 24% 짧음

- **충전 속도(C-rate)와 수명의 관계**
	- Batch 1: 0→80% 평균 C-rate가 높을수록 수명이 짧음 (Spearman ρ = −0.63, p < 0.001)
	- Batch 2·3: 모든 정책의 평균 충전 속도가 약 4.8C로 같아 평균 C-rate로는 비교 불가. 전환 SOC(Q1)가 크고 2단계 전류(C2)가 낮을수록 열화가 완만함
	- 핵심 발견 : Batch 2의 수명 차이는 C-rate보다 셀 구조(`newstructure`) 차이에서 옴

- **추가 확인**
	- 데이터 품질: Batch 1의 10셀은 EOL(0.88Ah)에 닿기 전에 기록이 끝나 `cycle_life`가 실제 수명보다 짧음 (중도절단). `cycle_life` 결측 10셀(Batch 2 VarCharge·SLOWCYCLE 8셀, Batch 3 EOL 미도달 2셀)은 제외
	- 다중공선성: ΔQ 계열(`log_var`, `log_abs_min`, `log_abs_mean`)은 서로 r ≥ 0.97, VIF 수백 ~ 수천
	- `chargetime_2_6`, IR, 온도는 배치마다 수명과의 상관 부호가 바뀜 → 배치 차이를 담은 피처

## Modeling

### 피처 엔지니어링 전략
- 모든 피처는 **사이클 2 ~ 100** 정보 또는 실험 전에 정해지는 충전 정책만 사용합니다. Knee, 후반 열화 속도처럼 100사이클 이후 정보는 누수라서 제외했습니다.
- 100사이클로 **먼저 자른 뒤** 스무딩합니다 (가운데 정렬 rolling median이 101사이클 이후 값을 참조하지 않도록).
- 후보 21개(ΔQ 통계 6 + summary 12 + 충전 정책 3)에서 피처 묶음 4개를 같은 조건으로 비교했습니다.

| 피처 묶음 | 내용 | 근거 |
|---|---|---|
| `logvar_only` | `log_var` | 원논문 Variance model, Day 1 기준선 |
| `day1_selected` | `log_var`, `kurtosis`, `Q1`, `slope_91_100` | Day 1: 세 배치의 상관 방향이 일관된 피처 |
| `b1_selected` | `log_var`, `slope_91_100`, `C1` | Day 2: Batch 1 train만으로 다시 선택 (테스트 배치 정답은 보지 않음) |
| `paper_discharge` | `log_abs_min`, `log_var`, `skew`, `kurtosis`, `Q2`, `Qmax_minus_Q2` | 원논문 Discharge model |

### 모델 선택 및 근거
- 데이터 분할 : Batch 1(중도절단 제외 36셀)을 충전 정책 단위로 나눔. Hold-out 20%(7셀)는 Valid, 나머지 29셀은 정책 GroupKFold(5)로 Train CV와 하이퍼파라미터 튜닝(nested). 같은 정책 셀이 train/valid에 나뉘면 정책을 외운 것이 성능으로 잡히기 때문입니다.
- 후보 모델 : Dummy(평균), Linear, ElasticNet, Gaussian Process, Random Forest, Gradient Boosting
- 최종 모델 : **ElasticNet + `paper_discharge` 6개 피처**
- 선택 이유 : 테스트를 보기 전에 정한 규칙으로 골랐습니다. 점수 = (Train CV MAPE + Hold-out seed 10회 평균 MAPE) / 2이고, 최저 점수와 0.5%p 안쪽이면 선형 계열을 우선합니다. GPR(6.50)과 ElasticNet(6.85)이 같은 수준이었고, Batch 2·3가 Batch 1 범위 밖이라 외삽이 가능한 선형 계열을 택했습니다.

## 성능 결과

| 구분 | | MAPE (%) | 비고 |
|---|---|---|---|
| Train (Batch 1 CV) | | 7.07 | 정책 GroupKFold(5), nested 튜닝, 29셀 |
| Valid (Batch 1 Hold-out) | | 6.26 | 정책 단위 20%, 7셀 (seed 10회 평균 6.63) |
| Test (Batch 2) | | 25.79 | 39셀 |
| | Gap (Train-Valid) | −0.81 | (+) : 과적합 의심 |
| | Gap (Valid-Test) | 19.53 | (+) : 배치간 일반화 저하 의심 |
| | **Gap (Target-Test)** | **16.69** | Target : 원논문 9.1% |
| Test (Batch 3) | | 14.67 | 44셀 (추가) |
| | Gap (Batch2-Batch3) | 11.12 | Test 성능 간 비교 |
| | Gap (Target-Test) | 5.57 | Batch 3 기준, 원논문 성능 비교 |

- Batch 1 안에서는 원논문(9.1%)보다 낮은 오차를 냈고 과적합 징후도 없지만, Batch 2에서는 원논문보다 16.69%p 높습니다.
- 보조 지표(`study_jiwon/results/metrics_by_set.csv`) : 최종 모델은 기준선(`log_var` 선형)보다 Batch 1(CV 7.07 vs 8.85)과 Batch 2(25.79 vs 28.56)에서 낫지만, Batch 3에서는 기준선이 더 낫습니다(12.81 vs 14.67).

## 오류 분석
- **Batch 2: 배치 전체를 길게 예측** — 기존 셀(+25.4%)과 `newstructure` 셀(+24.5%) 모두 비슷하게 과대 예측했습니다. EDA에서 본 "같은 ΔQ에서 Batch 2 수명이 약 24% 짧다"와 같은 크기입니다. 셀 5개의 실제 수명으로 오프셋만 보정해도 MAPE가 25.8% → 10.3%로 내려가므로, 셀 사이의 순서는 대체로 맞히고 있고 오차의 대부분은 배치 기준선 이동입니다.
- **Batch 3: 학습 범위 밖 장수명 셀을 짧게 예측** — 가장 크게 틀린 셀은 Batch 1 최대 수명(1,074)보다 긴 셀이었습니다 (예: 실제 1,642 → 예측 388). 이 셀들은 ΔQ 곡선 3.2 ~ 2.9V에 0보다 위로 솟는 봉우리가 있었고, 분산 피처(`log_var`)가 이 흔들림을 "열화가 큰 셀"로 읽었습니다. 배치마다 충전 커브 시작 시점이 달라 `Qdlin`을 단순 비교하면 생기는 왜곡으로 봅니다.
- **개선 방향** — 배치별 `Qdlin` 시작점을 맞추는 곡선 정렬 전처리, 여러 배치를 함께 학습하는 구조, 새 배치마다 일부 셀로 기준선을 보정하는 절차. 사후 실험으로 `log_var`를 빼면 Batch 1 성능은 그대로이고 Batch 3 MAPE가 14.67 → 11.27로 줄었지만, 테스트를 본 뒤의 아이디어라 후보 개선안으로만 남겼습니다.

## ESS 도메인 해석
- **활용**
	- 조기 수명 예측 → 교체 계획: 초기 100사이클(전체 수명의 약 5 ~ 25%)만으로 수명을 추정해, 교체 비용이 큰 ESS에서 셀 교체 시점과 예산을 미리 잡을 수 있습니다. 학습 배치와 같은 조건에서는 MAPE 6 ~ 7% 수준입니다.
	- 입고 셀 선별: 같은 로트 안에서 단수명 셀을 초기에 골라 팩 구성에서 뺄 수 있습니다.
	- 새 로트 보정: 조건이 다른 로트는 오차가 한쪽으로 쏠리므로, 일부 셀을 끝까지 시험해 기준선만 보정하는 운영 방식을 쓸 수 있습니다.
- **한계와 실 배포에 필요한 것**
	- 학습 배치와 충전 설계·셀 구조가 다른 배치에서는 오차가 크게 늘었습니다.
	- 이 데이터는 급속충전(3.6 ~ 8C) 가속 시험입니다. 실제 ESS는 1C 이하 저율, 부분 충방전, 계절별 온도 변화를 겪으므로 실제 운영 데이터로 재학습과 Live Test가 필요합니다.
	- 학습 데이터가 36셀로 적고, 학습 최대 수명(1,074)보다 긴 셀은 외삽에 의존합니다.
	- 입력 피처(특히 ΔQ 통계) 분포가 학습 때와 달라지면 경고하는 Data Drift 모니터링이 필요합니다.

## 참고문헌
- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391.

## 팀 구성
울산 4반
- 심혁 (U127) : EDA, 피처 엔지니어링, 모델 개발, 성능 평가(Batch 2·3) — `study_hyeock/`
- 허지원 (U139) : EDA, 피처 엔지니어링, 모델 개발, 성능 평가(Batch 2·3) — `study_jiwon/`
