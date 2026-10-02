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
│   ├── figures/                   # 두 노트북의 그림 (day1_F1 ~ F5: Q1~Q5 배치별, day1_D: 데이터 점검, day1_M: 설계 전략, day2_10-*: 오류 분석)
│   └── results/
│       ├── model_performance.csv  # 성능 표 (과제 Format)
│       ├── metrics_by_set.csv     # 보조 지표: RMSE · MAPE · bias, 기준선 대비
│       └── predictions.csv        # 셀별 예측값
│
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
│
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

Day 1 보고서(Q1 ~ Q5)의 순서를 따릅니다. 세 배치를 한 장에 모은 그림은 `study_hyeock/figures/`, 배치별 그림은 `study_jiwon/figures/`에 있습니다.

### Q1. Cycle Life 분포
- 중앙값 Batch 1 858 / Batch 2 472 / Batch 3 1,006. 단수명(< 500) 비율 0% / 72% / 0%, 장수명(> 1,000) 비율 22% / 8% / 52%
- 같은 실험 계열인데도 분포 위치가 다름. Batch 2는 Batch 1 범위의 왼쪽 바깥에 몰려 있고(쌍봉), Batch 3는 오른쪽 꼬리(1,300 ~ 1,935)가 Batch 1 범위를 넘어감
- 논문 분류 기준(550)으로 자르면 Batch 1의 단수명 클래스는 1셀뿐 → 분류는 학습 불가, 회귀 선택

<img src="study_hyeock/figures/F1-1_hist.png" width="49%"> <img src="study_hyeock/figures/F1-2_ratio.png" width="49%">

- **이상치** : Batch 2의 고수명 이상치 9셀은 전부 `newstructure` 셀로, 같은 충전 정책의 기존 셀보다 수명이 1.6 ~ 2.2배. Batch 1에는 `newstructure` 셀이 0개, Batch 3는 전부라 학습 데이터에는 이 효과가 없음
- **왜 짧은가** : Batch 1에서는 0→80% 평균 충전 속도가 빠를수록 수명이 짧고(r = −0.58), 수명 하위 5셀은 초기 100사이클 최고온도가 약 0.9°C 높음 (빠른 충전 → 발열 → 열화 가속)

<img src="study_hyeock/figures/F1-3_outlier.png" width="49%"> <img src="study_hyeock/figures/F1-4_why_short.png" width="49%">

- **중도절단** : Batch 1의 10셀은 EOL(0.88Ah)에 닿기 전에 기록이 끝나 `cycle_life`가 실제 수명보다 짧음 → 학습 시 제외(36셀)
- **log 변환** : 왜도 0.45 / 1.64 / 1.26 → 0.04 / 1.36 / 0.52. Batch 1·3는 거의 대칭이 되고, Batch 2는 쌍봉이라 남음 → Target = `log10(cycle_life)`

<img src="study_jiwon/figures/day1_F2-1_qd_curves_B1.png" width="49%"> <img src="study_hyeock/figures/F1-5_log.png" width="49%">

### Q2. 열화 곡선
- 세 배치 모두 한동안 평평하다가 어느 시점부터 급격히 떨어짐 (비선형)
- 100사이클 시점의 용량은 수명과 상관이 약함 (r = 0.32 / −0.28 / 0.34) → 용량 값보다 곡선 모양 기반 피처(ΔQ)가 필요함

<img src="study_hyeock/figures/F2-1_qd_curves.png" width="49%"> <img src="study_hyeock/figures/F2-4_early.png" width="49%">

- 수명의 약 60%까지는 0.005 Ah / 100 cycle 안팎으로 거의 일정하다가 이후 급격히 가속 (마지막 20% 구간 속도 = 전체 평균의 약 3.4배, Batch 2가 가장 가파름)
- Knee point는 세 배치 모두 수명의 약 75 ~ 79% 지점. 다만 100사이클 이후 정보라 피처로 쓰면 누수 → 해석용으로만 사용

<img src="study_hyeock/figures/F2-2_rate.png" width="49%"> <img src="study_hyeock/figures/F2-3_knee.png" width="49%">

### Q3. ΔQ(V) 곡선
- ΔQ(V) = Q100(V) − Q10(V) (`Qdlin[99] - Qdlin[9]`, 인덱스 = 사이클 − 1). 3.2V 근처에서 꺼지고 2.9 ~ 3.0V에서 가장 깊어짐
- 골이 생기는 전압 위치는 장수명/단수명이 같고, 차이는 골의 깊이와 폭 (단수명이 장수명 대비 1.5 ~ 2.5배 깊음)

<img src="study_hyeock/figures/F3-1_dq_curves.png" width="49%"> <img src="study_hyeock/figures/F3-2_long_short.png" width="49%">

- `log_var` = log10 var(ΔQ)가 세 배치 모두 가장 강한 피처 (r = −0.87 / −0.92 / −0.76)
- 다만 Batch 1으로 맞춘 직선 기준 Batch 2는 수명이 약 24% 짧음 → Batch 1만으로 학습하면 Batch 2를 과대 예측할 가능성

<img src="study_hyeock/figures/F3-3_logvar.png" width="60%">

- ΔQ 통계 피처 6개 중 크기 계열(`log_var`, `log_abs_min`, `log_abs_mean`)은 일관되게 강하고, `skew`·`kurtosis`는 배치마다 부호와 크기가 다름. Batch 2의 −0.92는 `newstructure` 두 군집 차이로 부풀려졌을 가능성

<img src="study_jiwon/figures/day1_F3-3_dq_feature_corr_B1.png" width="32%"> <img src="study_jiwon/figures/day1_F3-3_dq_feature_corr_B2.png" width="32%"> <img src="study_jiwon/figures/day1_F3-3_dq_feature_corr_B3.png" width="32%">

### Q4. 충전 조건(C-rate)과 수명
- 프로토콜별 평균 수명 : Batch 1은 저속(3.6C · 4C · 4.4C)이 약 1,075 ~ 1,230으로 가장 길고 5.4C(80%)-5.4C, 7C·8C 계열이 짧음. Batch 2는 `newstructure` 프로토콜만 약 870 ~ 990이고 나머지는 약 400 ~ 480

<img src="study_jiwon/figures/day1_F4-1_protocol_life_B1.png" width="32%"> <img src="study_jiwon/figures/day1_F4-1_protocol_life_B2.png" width="32%"> <img src="study_jiwon/figures/day1_F4-1_protocol_life_B3.png" width="32%">

- 고속 충전 → 수명 단축은 Batch 1에서만 유의 (Spearman ρ = −0.63, p < 0.001). Batch 2·3는 모든 정책의 평균 충전 속도가 약 4.8C로 같아 비교 불가 (C1 기준 p = 0.739 / 0.135)

<img src="study_jiwon/figures/day1_F4-2_fast_charge_B1.png" width="32%"> <img src="study_jiwon/figures/day1_F4-2_fast_charge_B2.png" width="32%"> <img src="study_jiwon/figures/day1_F4-2_fast_charge_B3.png" width="32%">

- 평균 속도가 고정된 조건에서는 전환 SOC(Q1)가 크고 2단계 전류(C2)가 낮을수록 열화 곡선이 완만함
- 핵심 발견 : Batch 2의 수명 차이는 C-rate보다 셀 구조(`newstructure`) 차이에서 옴

<img src="study_jiwon/figures/day1_F4-3_charge_vs_degradation_B1.png" width="32%"> <img src="study_jiwon/figures/day1_F4-3_charge_vs_degradation_B2.png" width="32%"> <img src="study_jiwon/figures/day1_F4-3_charge_vs_degradation_B3.png" width="32%">

### Q5. 상관관계와 다중공선성
- ΔQ 계열이 세 배치 모두 상위권으로 가장 안정적인 신호. 초기 용량(`Q2`)은 상관이 약함 (0.10 / −0.20 / 0.17)
- `chargetime_2_6`, IR, 온도는 배치마다 수명과의 상관 부호가 바뀜 (예: `chargetime_2_6` +0.72 / −0.94 / +0.60) → 배치 차이를 담은 피처

<img src="study_jiwon/figures/day1_F5-1_feature_corr_B1.png" width="32%"> <img src="study_jiwon/figures/day1_F5-1_feature_corr_B2.png" width="32%"> <img src="study_jiwon/figures/day1_F5-1_feature_corr_B3.png" width="32%">

- Batch 2의 1위 `chargetime_2_6`(r = −0.94)은 `newstructure` / 기존 셀 두 군집으로 완전히 나뉘고 군집 안에서는 추세가 없음

<img src="study_jiwon/figures/day1_F5-2_top3_B1.png" width="32%"> <img src="study_jiwon/figures/day1_F5-2_top3_B2.png" width="32%"> <img src="study_jiwon/figures/day1_F5-2_top3_B3.png" width="32%">

- ΔQ 계열(`log_var`, `log_abs_min`, `log_abs_mean`)은 서로 r ≥ 0.97, VIF 수백 ~ 수천 → 대표 피처만 선택하거나 규제 모델 필요

<img src="study_jiwon/figures/day1_F5-3_multicollinearity_B1.png" width="32%"> <img src="study_jiwon/figures/day1_F5-3_multicollinearity_B2.png" width="32%"> <img src="study_jiwon/figures/day1_F5-3_multicollinearity_B3.png" width="32%">

### 데이터 점검
- `cycle_life` 결측 10셀(Batch 2 VarCharge·SLOWCYCLE 8셀, Batch 3 EOL 미도달 2셀)은 타깃이라 보간하지 않고 제외
- 물리적으로 불가능한 값(용량 1.1Ah 셀의 2Ah 방전, Tmax 400°C, IR = 0 등)은 기록 오류로 보고 제거. 수명 말기의 용량 급락·IR 상승은 실제 열화라 유지

<img src="study_jiwon/figures/day1_D_outliers_by_cycle.png" width="80%">

### Day 1 모델 설계
- 피처 선별 기준 : ① 누수 금지(사이클 2 ~ 100 또는 실험 전 값) ② 세 배치의 상관 부호가 같고 가장 약한 배치에서도 |r| ≥ 0.15 ③ |r| ≥ 0.7 중복 제거 → `log_var`, `kurtosis`, `Q1`, `slope_91_100` (VIF ≤ 1.3)

<img src="study_jiwon/figures/day1_M1-1_feature_consistency.png" width="49%"> <img src="study_jiwon/figures/day1_M1-2_selected_features.png" width="49%">

- Target = `log10(cycle_life)` : skew 0.89 → −0.09. 평가는 `10 ** ŷ`로 되돌려 사이클 단위로 계산

<img src="study_jiwon/figures/day1_M2_target_distribution.png" width="70%">

## Modeling

### Day 1 설계 → Day 2 변경점
Day 1 설계(Target, 누수 방지, 정책 GroupKFold, 기준선, 후보 모델)는 그대로 따랐고, 아래 세 가지를 바꿨습니다.

| 항목 | Day 1 설계 | Day 2 실제 | 바꾼 이유 |
|---|---|---|---|
| 학습 데이터 | 세 배치 합쳐서 학습(129셀), 배치 단위 LOBO로 보조 검증 | Batch 1(중도절단 제외 36셀)만 학습, Batch 2·3는 테스트 | 과제 형식(Train = Batch 1, Test = Batch 2). Day 1에서 걱정한 "처음 보는 배치" 상황을 테스트로 직접 확인 |
| 피처 선별 기준 | 세 배치 모두 상관 부호가 같은 피처 | Batch 1 train만 보고 다시 선택(`b1_selected`) | Day 1 기준은 Batch 2·3의 실제 수명을 보고 계산하므로, Batch 1만 학습하는 구조에서는 테스트 정보 누수 |
| 최종 피처 | `log_var`, `kurtosis`, `Q1`, `slope_91_100` | `paper_discharge` 6개 | 피처 묶음 4개를 같은 규칙으로 비교한 결과 (아래). Day 1에서 뺀 `Q2`, `skew`, `log_abs_min`이 다시 들어갔지만, ElasticNet이 `skew`·`log_abs_min` 계수를 0으로 만들어 실제로는 `log_var` 중심 |

`day1_selected` 묶음도 같은 조건으로 비교 후보에 넣어, Day 1 선별 결과가 Day 2에서 어떤 성능을 내는지 함께 확인했습니다.

### 피처 엔지니어링 전략
- 모든 피처는 **사이클 2 ~ 100** 정보 또는 실험 전에 정해지는 충전 정책만 사용합니다. Knee, 후반 열화 속도처럼 100사이클 이후 정보는 누수라서 제외했습니다.
- 100사이클로 **먼저 자른 뒤** 스무딩합니다 (가운데 정렬 rolling median이 101사이클 이후 값을 참조하지 않도록).
- 후보 21개(ΔQ 통계 6 + summary 12 + 충전 정책 3, Day 1의 22개에서 배치와 섞인 `newstructure` 제외)에서 피처 묶음 4개를 같은 조건으로 비교했습니다.

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

<img src="study_jiwon/figures/day2_10-5_coefficients.png" width="55%">

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

<img src="study_jiwon/figures/day2_10-1_parity.png" width="80%">

## 오류 분석
- **Batch 2: 배치 전체를 길게 예측** — 기존 셀(+25.4%)과 `newstructure` 셀(+24.5%) 모두 비슷하게 과대 예측했습니다. EDA에서 본 "같은 ΔQ에서 Batch 2 수명이 약 24% 짧다"와 같은 크기입니다. 셀 5개의 실제 수명으로 오프셋만 보정해도 MAPE가 25.8% → 10.3%로 내려가므로, 셀 사이의 순서는 대체로 맞히고 있고 오차의 대부분은 배치 기준선 이동입니다.
- **Batch 3: 학습 범위 밖 장수명 셀을 짧게 예측** — 가장 크게 틀린 셀은 Batch 1 최대 수명(1,074)보다 긴 셀이었습니다 (예: 실제 1,642 → 예측 388). 이 셀들은 ΔQ 곡선 3.2 ~ 2.9V에 0보다 위로 솟는 봉우리가 있었고, 분산 피처(`log_var`)가 이 흔들림을 "열화가 큰 셀"로 읽었습니다. 배치마다 충전 커브 시작 시점이 달라 `Qdlin`을 단순 비교하면 생기는 왜곡으로 봅니다.

<img src="study_jiwon/figures/day2_10-2_error_by_group.png" width="80%">

<img src="study_jiwon/figures/day2_10-4_batch3_dq_peak.png" width="80%">
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
