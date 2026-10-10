# PINNecho 프로젝트 작업 상태

> 최종 업데이트: 2026-09-20 · 브랜치 `cursor/fsi-pinn-doppler-stage1-935e` · PR #1
> 테스트: **130 passing** (`pytest`)

FSI 정보 기반 물리정보신경망(PINN)으로 희소·단일성분 도플러 측정에서 좌심실 내부
유동장(속도·압력·와도·잔류시간)을 복원하고, 두 물리 백본을 비교하는 프로젝트의
현재 상태를 정리한 문서입니다.

---

## 1. 목표와 핵심 질문

- **입력**: 희소하고 잡음이 있으며 **빔 방향 성분만** 측정되는 도플러 유사 속도 데이터.
- **출력**: 전체 속도장, 압력, 와도, 벽 전단응력(WSS), (실데이터 확보 시) 잔류시간.
- **핵심 과학 질문(원안)**: FSI 정보로 **압력·구배량** 복원이 개선되는가? → **압력에 관한 한
  주장 기각.** 대조 실험 결과 압력 이득은 경계 압력 주입(순환, `corr(f,∇p)=0.95`) + 관측 부족의
  하류 증상임이 확인됨.
- **구배량 대체 주장도 기각**: forcing 셔플 진단(Ablation 6, 5시드) 결과, 구배 이득은 일반적
  물리 정칙화가 아니라 **궤적-특이 정보(비선형 누출)**이며, 공정한 동일-창 비교에선 WSS 우위도
  유의하지 않음. manufactured 설정에선 forcing 기반 이득(압력·구배 모두)이 본질적으로 궤적
  정보를 담는다.
- **밴드-국소 오라클 dry-run(Ablation 7, 3시드)**: forcing을 실제처럼 벽 밴드로 국소화(공동의
  48%만 덮음, 내부 B=A)하면 압력 이득이 사라지는 정도가 아니라 **음(-)**이 됨(relL2 1.06→13.3,
  0/3) — full-cavity "압력 이득"은 forcing≈∇p 순환의 산물이었음을 합성 상한에서 확인. 속도
  이득은 미미(Δ0.022)하고 support-preserving 셔플·band-mask 대조에서 baseline으로 회귀. gate R
  후 실 export에 그대로 돌릴 `A/exact/shuffle/band_mask` 분석을 **사전 등록**.
- **순환성과 독립적으로 남는 견고한 결과**: **관측성(다중 음향창)** — FSI 없는 baseline에서도
  교차빔 `u`가 창 수에 단조 반응(0.98→0.75), SNR·희소성은 거의 무관. 단, 다중창이 주 성분 `v`·
  WSS corr는 오히려 낮출 수 있음. 전용 정리: [`docs/OBSERVABILITY.md`](OBSERVABILITY.md).
- **3D 평면-커버리지 관측성**(표준 에코 뷰, FSI 무관): 성분별로 커버리지 제약이 드러남 —
  축방향 `w`는 단일 심첨 평면만으로도 먼저 복원(0.954), 측방 `u`는 상보적 프로브 위치
  (parasternal)를 더해야 관측 가능(0.996→0.975), 측방 `v`는 표준 A4C/A2C/PLAX/PSAX
  세트로는 구조적으로 관측 불가(≈0.99)—이상적 전체-체적 창에서만 복원 시작(0.978).
  절대오차는 CPU 예산에서 compute-limited이나 성분 순서는 시드 잡음 위. 동 문서에 수록.
- **커버리지 공백 닫기**: `y`빔을 가진 비표준 창(`lat_y`) 하나를 추가하면 측방 `v`가
  0.995→**0.972**로 떨어져 이상적 전체-체적 기준(0.978)에 도달 — 즉 측방-`y` 사각지대는
  재구성 방법의 한계가 아니라 **순수 빔-방향(커버리지) 공백**이며 상보 창 하나로 닫힘.
- **Stage 1 범위**: 실제 환자·에코 데이터 없음. IBAMR/IBFE 파이프라인 출력을 모사한
  **해석적·자기일관(divergence-free, exact no-slip, NS-exact) 합성 지상진값** 위에서 검증.
- **데이터 출처(중요)**: PINNecho가 받는 모든 양은 **전진 FSI 시뮬레이션 출력**이다. MRI
  (HVSMR-2.0, ECG-게이트·정지 3D CMR)는 **LV 기하만** 제공하고(심근 ~3 mm 쉘·무응력 기준
  *가정*), 벽 운동·속도는 주지 않는다. 즉 벽속도·압력·forcing은 모두 시뮬레이션 값이며 **측정
  속도 지상진값은 없다**. 프로젝트 전체에서 유일한 실측은 최종 임상 **도플러 에코**뿐이다.

### 두 백본 (아키텍처는 동일, 물리항만 다름)

| | A — `baseline` | B — `fsi_informed` |
|---|---|---|
| 운동량 forcing `f` | `0` (일반 비압축 NS) | IBAMR/IBFE FSI 체적력 (**오라클 전용**) |
| 벽 경계조건 | 운동학적 no-slip (윤곽 추적 속도) | 정확한 FSI 벽속도 + (옵션) traction 연속성 |

> **오라클 vs 실제 전달물.** 실 IBFE forcing `f`는 IB 커널로 퍼진 구조 라그랑지안
> 힘이라 **벽에서 ~3셀 밴드 안에서만 0이 아니고 공동 내부는 정확히 0**이다(밴드 밖에선
> Model B = Model A). 또한 `f`·traction·압력은 **평가용 지상진값(오라클)일 뿐 실제 Model
> B 입력이 아니다** — 임상 에코에선 얻을 수 없다. 전달 가능한 Model B의 입력은 **벽
> 운동학(벽속도) + 저차원 활성화 파라미터(`T_max`, 타이밍 `g(t)`)**뿐이고, `f`를 학습에
> 넣는 것은 "forcing을 알았다면" 상한을 재는 **오라클 실험**이다.
>
> **전달형 Model B(`fsi_param`) 구현됨.** 그 전달형 입력 조건(`f` 미접근 + 저차원 활성화
> ansatz `f = T_max·g(t)·w(x)·d̂(x)`, 자유도 5개; `models/activation_forcing.py`)을 세 번째
> 백본 `fsi_param`으로 구현했다. 밴드-국소 합성 상한에서 전달형 B는 **Model A와 구별되지
> 않으며**(Ablation 8), 오라클에서만 나타나는 미미한 속도 이득은 전달형이 **접근할 수 없는**
> 양이다. 즉 `fsi_informed`는 오라클(참 `f`) 상한, `fsi_param`은 임상 전달물이다.

---

## 2. 한눈에 보는 진행 상태

| 단계 | 내용 | 상태 |
|---|---|---|
| Stage 1 | 물리 잔차·네트워크·손실·합성 지상진값·Model A 학습 | ✅ 완료 |
| Stage 1 | Model B(FSI forcing + FSI 벽 + traction) A/B 비교 | ✅ 완료 |
| Stage 2 | 정식 도플러 합성기(희소/에일리어싱/SNR), 벽속도 분기, IBFE 인터페이스 | ✅ 완료 |
| 도구화 | A/B 시각화·애니메이션, 관측성 스윕, float32/체크포인트, 콘솔 스크립트·CI | ✅ 완료 |
| 3D | 체적보존 타원체 합성 지상진값 + 3D 엔드투엔드 학습 경로 | ✅ 검증(정확도는 예산 한계) |
| 데이터 준비 | 실제 IBFE 수용 파이프라인(I/O·검증·어댑터·3D/밸브 exporter·가이드) | ✅ 완료 |
| 실데이터 | 실제 IBAMR/IBFE export 연결, 잔류시간 실측 유입, 고해상도 3D | ⏳ 데이터 대기 |

---

## 3. 컴포넌트별 구현 상태

| 컴포넌트 | 파일 | 상태 |
|---|---|---|
| 비압축 NS 잔차 (2D·3D, forcing 훅) | `physics/ns_residual.py` | ✅ + 단위테스트(Taylor–Green, Poiseuille) |
| 잔류시간 스칼라 전달 | `physics/scalar_transport.py` | ✅ + 단위테스트 |
| autograd 연산자(grad/div/curl/lap) | `physics/operators.py` | ✅ |
| 공유 네트워크(멀티스케일 Fourier, `u,v,[w],p,c`) | `models/mlp_pinn.py` | ✅ 2D/3D, 체크포인트 저장/로드 |
| 복합손실(6항 + traction, 어닐링) | `train/composite_loss.py` | ✅ 데이터항은 빔 성분만 |
| Model A 운동학 BC / 밸브 / 유입 | `bc/boundary_conditions.py` | ✅ |
| Model B FSI 벽속도 + traction 연속성 | `bc/boundary_conditions.py` | ✅ + 단위테스트 |
| 평가 지표(속도/압력/와도/Q/WSS/잔류시간 + 스플라인 기준선) | `evaluate.py` | ✅ |
| 2D 합성 지상진값 | `data/synthetic_lv.py` | ✅ (div~1e-15, no-slip~1e-16, NS-exact) |
| 3D 합성 지상진값(체적보존 타원체) | `data/synthetic_lv_3d.py` | ✅ + 단위테스트 |
| 정식 도플러 합성기(희소/에일리어싱/SNR) | `data/synthesize_doppler.py` | ✅ + 단위테스트 |
| **다중-평면 음향창 지오메트리**(A4C/A2C/PLAX/PSAX 슬랩 선택) | `data/acquisition.py` | ✅ + 단위테스트 |
| A/B 시각화(필드 패널 + 심장주기 GIF) | `eval/visualize.py` | ✅ + 스모크 |
| 관측성 스윕(윈도우/SNR/희소성) | `train/observability.py` | ✅ + 스모크 |
| **3D 평면-커버리지 관측성 스윕**(A4C/A2C/PLAX/PSAX vs 이상적 점-윈도우) | `train/plane_coverage.py` | ✅ + 스모크 |
| **밴드-국소 forcing 오라클 스윕**(A/exact/shuffle/band_mask) | `train/band_oracle.py` | ✅ + 스모크 |
| **전달형 Model B 저차원 활성화 forcing**(`f = T_max·g(t)·w(x)·d̂(x)`, 자유도 5) | `models/activation_forcing.py` | ✅ + 단위테스트 |
| **전달형 3-way 스윕**(A / `fsi_param` 전달물 / 오라클 상한) | `train/band_oracle.py --conditions deliverable` | ✅ + 스모크 |
| **벽-트래킹 노이즈 프리셋**(`exact` / `placeholder` / `ste` 문헌 보정) | `data/tracking_noise.py` | ✅ + 단위테스트 |
| **공개 4D-flow / phantom 볼륨 로더**(`f = 0`, 오라클 아님) | `data/public_volume.py` | ✅ + 단위테스트 |
| **공개 볼륨 A vs `fsi_param` 스윕** | `train/public_volume_ab.py` | ✅ + 스모크 |
| 3D 엔드투엔드 드라이버 | `train/train3d.py` | ✅ + 스모크 |
| **실 IBFE I/O**(NPZ·매니페스트/CSV·VTK) | `data/ibfe_io.py` | ✅ + 단위테스트 |
| **IBFE 검증기** | `data/ibfe_validate.py` | ✅ + 단위테스트 |
| **IBFE→학습 어댑터** | `data/ibfe_dataset.py` | ✅ + 단위테스트 |
| IBFE 로더 분기 + 3D/밸브 합성 exporter | `data/load_ibfe_output.py` | ✅ (실 파일 로딩 활성) |

---

## 4. 핵심 결과

### 4.1 Model A vs Model B — 그리고 대조 실험으로 드러난 교란요인 ⚠️

초기 비교(벽속도 분기, 동일 아키텍처/데이터/초기화)는 큰 압력 개선(상관 0.24 → 0.996)을
보였으나, **이 결과는 그대로 논문 근거로 쓸 수 없다.** 두 교란요인이 있다:

1. **순환성** — 합성 지상진값은 manufactured solution이라, Model B에 주는 traction
   `−p·n + 점성항`이 **해석적 압력 `p`를 벽에서 직접 주입**하는 셈. 압력 복원은 거의 당연.
2. **비대칭 경계정보** — 초기 비교에서 A는 노이즈 낀 벽, B는 정확한 벽을 받았다.

**대조 실험(양쪽 정확 벽, 3시드 평균)으로 물리항 순수 효과를 격리한 결과:**

| variant | pressure relL2 | pressure corr | vorticity corr | WSS corr |
|---|---|---|---|---|
| `A_exact` (정확 벽, forcing 없음) | 1.134 | −0.039 | 0.740 | 0.640 |
| `B_forcing` (정확 벽 + forcing, traction 없음) | 3.900 | 0.641 | 0.782 | 0.659 |
| `B_forcing_traction` (+ exact traction) | 0.171 | **0.986** | 0.776 | 0.655 |

- **압력**: forcing 단독으로는 크기 복원 실패(relL2 3.9). corr 0.986은 **오직 traction 주입**으로
  달성 → 초기 0.996은 "FSI 이득"이 아니라 경계 압력 주입 산물(≈순환적).
- **와도·WSS**: forcing 순효과는 `A_exact` 대비 **+0.04 / +0.02**로 소폭(시드 산포 수준).
  기존에 커 보인 WSS 개선폭은 대부분 **noisy vs exact 벽** 차이였다.
- **traction 섭동 스트레스 테스트**: ε=20%에도 pressure corr 0.98 유지(스케일 불변 지표라
  둔감), 크기 지표 relL2만 0.17→0.21(~24%) 악화.

**추가 대조 실험(point 1·2 검증):**
- **Check 0 (순환성 게이트, 학습 불필요)**: `corr(forcing, ∇p)=0.95`, ∇p가 forcing 크기의 94% →
  현 forcing은 사실상 압력 그라디언트. **Check 0b(vorticity 게이트)**: `corr(f, μ∇×ω)≈0`,
  점성=vorticity-curl 항은 forcing의 **0.23%뿐**(관성 지배 혈류) → **vorticity-leak 가설은
  반증**됨. 재설계 검증은 두 게이트(`--which coupling`).
- **Ablation 3 (압력=관측 부족의 하류 증상)**: baseline+정확 벽에서 음향창 1→3으로 늘리면 교차빔
  `u` 0.96→0.67, pressure corr −0.04→0.30 동반 개선(결합 −0.67). 압력 실패는 "물리항 부재"가
  아니라 단일창 관측 부족 탓.
- **Ablation 4 (관측 vs 물리 대체, 비순환)**: 속도/교차빔은 2번째 창이 압도(대체 불가)하나,
  구배량은 `w1_forcing`이 `w2_baseline`을 상회.
- **Ablation 5 (forcing 섭동, 3시드)**: 구배 이득이 30% 섭동에 강건해 보였으나, 기준선이 2창
  baseline이고 n=3이라 취약 → Ablation 6에서 교정됨.
- **Ablation 6 (forcing 셔플 진단, 5시드) — 가장 결정적 실험**: 궤적-무관(순열) forcing은 이득을
  못 지키고 오히려 해침(`shuffled−baseline` 1/5). `exact>shuffled` → **구배 이득은 궤적-특이
  비선형 누출**(선형 상관 게이트가 못 잡음). 통계는 t-test + **부호검정** 병기: **WSS는 무효**
  (동일 1창 t=0.70, 2/5, 부호 p=0.81; Ablation 5의 "유의"는 2창-baseline 교란), **vorticity는
  방향 일관**(5/5, 부호 p=0.031)이나 t-test 크기 확정엔 검정력 부족 — 죽은 게 아니라 "미확정".
- 전체 표·해석·권고: **[`docs/ABLATIONS.md`](ABLATIONS.md)**, 원자료
  `docs/results/ablations_all.json`(Check0+1~4) · `docs/results/ablation5.json` ·
  `docs/results/ablation6.json` · `docs/results/ablations.json`(초기 2종).
- (참고) 초기 단일시드 비교 원자료: `docs/results/stage2_forcing.json`,
  `docs/results/stage2_traction.json`.

### 4.2 관측성 스윕 — **FSI 무관 독립 결과** (baseline 백본, FSI/traction 없음, 5시드)

전용 문서: **[`docs/OBSERVABILITY.md`](OBSERVABILITY.md)**. 순환성 논쟁과 완전히 독립(FSI 항 미사용).

| 조건 | 윈도우 | SNR(dB) | pts | `u` relL2 | `v` relL2 | vort corr | WSS corr | p corr |
|---|---|---|---|---|---|---|---|---|
| windows=1 | 1 | 26 | 300 | **0.982±0.004** | 0.718 | 0.582 | 0.428 | −0.154 |
| windows=2 | 2 | 26 | 300 | **0.855±0.036** | 0.760 | 0.565 | 0.271 | −0.032 |
| windows=3 | 3 | 26 | 300 | **0.749±0.040** | 0.811 | 0.598 | 0.205 | 0.241 |
| noise=0.02 | 1 | 34 | 300 | 0.983 | 0.720 | 0.579 | 0.423 | −0.118 |
| noise=0.10 | 1 | 20 | 300 | 0.982 | 0.720 | 0.576 | 0.426 | −0.115 |
| points=150 | 1 | 26 | 150 | 0.986 | 0.701 | 0.579 | 0.417 | −0.028 |
| points=600 | 1 | 26 | 600 | 0.978 | 0.695 | 0.575 | 0.405 | 0.007 |

- **지배 변수는 각도(윈도우 수)**: 교차빔 `u`가 0.982→0.855→**0.749**로 단조 감소(시드 산포 밖).
  압력 corr도 −0.15→−0.03→**+0.24**로 동반 회복(Ablation 3 커플링).
- **SNR(20–34dB)·희소성(150–600)은 `u`를 <1% 변동** → 병목은 명확히 **관측 방향 수**.
- **주의(공짜 아님)**: 창을 늘리면 주 관측 성분 `v`와 **WSS corr는 오히려 저하**(0.43→0.21).
  다중창은 *미관측 방향·전역 압력*을 되찾을 뿐 모든 파생량에 유리하진 않음.
- 원자료: `docs/results/observability/sweep.json`.

### 4.3 3D 엔드투엔드

- 지상진값 자기일관성: `max|div u|`~1e-15, 벽 no-slip~1e-12, forcing 포함 운동량 잔차=0.
- 전체 파이프라인(차원 일반화 잔차/네트워크/손실/트레이너)이 3D에서 학습 진행(손실 감소).
- 정확도는 관측성·CPU 예산 한계(짧은 런에서 ~0.8 relL2) → 예산·다중 윈도우·실 3D 지상진값이 향후 과제.

### 4.4 시각화 산출물

`docs/results/ab_viz/`: 진실/A/B 필드 패널(`fields_t0.30.png`, `fields_t0.60.png`),
지표 막대(`metric_bars.png`), 심장주기 애니메이션(`cycle_speed.gif`, `cycle_vorticity.gif`).
패널에서 Model B가 baseline이 뭉개는 **압력 구배·와도 구조**를 복원함을 육안 확인.

---

## 5. 실행 진입점

콘솔 스크립트(`pip install -e ".[dev,viz]"` 후) 또는 `scripts/`에서 직접 실행:

| 목적 | 콘솔 스크립트 | 스크립트 |
|---|---|---|
| Model A 학습(토이 2D) | `pinnecho-train-a` | `scripts/train_model_a.py` |
| A/B 비교 | `pinnecho-compare-ab` | `scripts/compare_models_ab.py` |
| A/B 시각화 + 애니메이션 | `pinnecho-visualize` | `scripts/visualize_ab.py` |
| 관측성 스윕 | `pinnecho-sweep` | `scripts/observability_sweep.py` |
| 3D 평면-커버리지 스윕 | `pinnecho-coverage` | `scripts/plane_coverage_sweep.py` |
| 밴드-국소 forcing 오라클 스윕 | `pinnecho-band-oracle` | `scripts/band_oracle_sweep.py` |
| 전달형 Model B 3-way 스윕 | `pinnecho-band-oracle --conditions deliverable` | `scripts/band_oracle_sweep.py --conditions deliverable` |
| 3D 검증 | `pinnecho-train-3d` | `scripts/train_model_3d.py` |
| 예제 IBFE export 생성 | — | `scripts/make_example_ibfe_export.py` |
| 예제 공개 볼륨 NPZ | — | `scripts/make_example_public_volume.py` |
| 공개 볼륨 A vs 전달형 B | `pinnecho-public-ab` | `scripts/public_volume_ab.py` |

공통 옵션: `--dtype {float32,float64}`, `--steps`, `--lbfgs-iters`, `--seed`.

---

## 6. 테스트 현황 (130 passing)

| 파일 | 검증 대상 |
|---|---|
| `test_operators.py` | autograd 미분 연산자 vs 해석해 |
| `test_ns_residual_taylor_green.py` | NS 잔차 ~0 (Taylor–Green, Poiseuille) |
| `test_scalar_transport.py` | 스칼라 전달 잔차 ~0 |
| `test_traction.py` | 유체 traction + 연속성 (2D/3D) |
| `test_synthetic_lv.py` / `test_synthetic_lv_3d.py` | 2D/3D 합성 지상진값 자기일관성 |
| `test_doppler.py` / `test_synthesize_doppler.py` | 도플러 투영·잡음·에일리어싱·SNR |
| `test_acquisition.py` | 다중-평면 음향창 지오메트리(A4C/A2C/PLAX/PSAX)·슬랩 선택·엔드투엔드 |
| `test_ibfe_export.py` / `test_ibfe_pipeline.py` | IBFE 인터페이스 + I/O·검증·어댑터 (2D/3D 왕복·학습) |
| `test_visualize_smoke.py` / `test_observability_smoke.py` | 시각화·스윕 배관 |
| `test_plane_coverage_smoke.py` | 3D 평면-커버리지 스윕 배관(프로토콜·지표·커버리지) |
| `test_train_model_a_smoke.py` / `test_train_3d_smoke.py` | Model A/B·3D 학습 스모크 |
| `test_ablation_smoke.py` | 물리항 격리·traction 섭동 대조 실험 배관 |
| `test_forcing_oracle.py` | 밴드-국소 forcing·support-preserving 셔플·band-mask 대조·검증기 밴드율 |
| `test_band_oracle_smoke.py` | 밴드-국소 합성 forcing 옵션·커버리지 리포터·A/exact/shuffle/band_mask 스윕 |
| `test_deliverable_model_b.py` | 전달형 Model B(`fsi_param`): 저차원 활성화 모듈·기하 밴드 템플릿·백본 배선·3-way 전달형 스윕 |
| `test_tracking_noise.py` | 벽-트래킹 프리셋(`exact`/`placeholder`/`ste`)·레거시 공식 재현·IBFE Model A 경로 |
| `test_public_volume.py` | 공개 4D-flow/phantom 볼륨→IBFEFrames(`f=0`)·NPZ 분기·A vs `fsi_param` 스윕 |
| `test_config.py` / `test_pipeline_smoke.py` | 설정 왕복·엔드투엔드 스모크 |

CI: `.github/workflows/ci.yml`가 Python 3.10/3.11에서 `pytest` 전체 실행.

---

## 7. 실 IBAMR/IBFE 데이터 준비 상태

실제 export 연결에 **코드 수정 불필요** — 디스크 계약만 맞추면 됨. 상세: [`docs/DATA_PREPARATION.md`](DATA_PREPARATION.md).

- **포맷**: 단일 **NPZ 번들** 또는 **`manifest.yaml` + 프레임별 CSV**(템플릿
  `configs/ibfe_manifest.template.yaml`), 선택적 VTK/Exodus(`meshio`).
  `load_ibfe_output(path)`가 경로로 자동 분기(`time_range`/`subsample` 지원).
- **검증**: `validate_ibfe_frames(frames)` — shape·유한성·법선·시간·물리량 점검.
- **학습**: `train_ibfe(frames, backbone=…, use_traction=…, predict_scalar=…)` →
  기존 네트워크+복합손실로 바로 학습, `evaluate_ibfe`로 홀드아웃 평가.
- **forcing 오라클 진단(실데이터)**: 실 IBFE forcing은 **밴드-국소**(공동 내부 0 ⇒ 밴드 밖
  B=A)이고 u에서 독립적이지도 않다(구조가 u를 따라 움직임·밴드 운동량 균형 폐합·커널 지지가
  벽 위치를 표시). 따라서 평가용 **오라클 전용**으로만 쓰며, `train_ibfe(..., forcing_control=
  "shuffle")`(밴드 내부만 섞는 **support-preserving** 셔플, 내부 0 유지)와 `forcing_control=
  "band_mask"`(크기×내향 법선, 기하 지지만 유지) **두 대조**를 함께 돌린다. 두 대조를 모두
  이기고 남는 이득만 "순수 물리 효과"의 상한으로 해석 가능(그래도 전달물 Model B는 `f`를
  쓰지 않으므로 오라클 상한일 뿐).
- **예제**: `python scripts/make_example_ibfe_export.py --dim {2,3} [--valve]`.

필드 요약: 유체 볼륨(`coords/velocity/pressure/forcing_fluid`), 인터페이스
(`coords/normals/velocity/traction_wall`), 밸브(`coords/velocity_mitral|aortic`),
메타(`cycle_period, rho, mu, spatial_dim`). 단위는 SI, 좌표는 공간→시간 순.

---

## 8. 남은 작업 (데이터 의존)

1. **실제 IBAMR/IBFE export 연결** — 위 계약에 맞춰 `load_ibfe_output(path)`로 투입.
   배관·검증·학습 경로는 모두 구현·테스트 완료. 공개 4D-flow / phantom은
   **오라클이 아닌** 별도 경로(`data/public_volume.py`, `f = 0`)로 이미 연결됨 —
   [`docs/PUBLIC_VOLUME.md`](PUBLIC_VOLUME.md).
2. **잔류시간 실측 복원** — 실제 승모판 유입 필드(또는 개방형 공동 합성) 필요.
   스칼라 잔차·`c` 헤드·유입 재초기화·`with_valve` 유입 점군은 완비.
3. **고해상도 3D** — 더 큰 학습 예산 + 다중 윈도우 취득 + 실 3D 지상진값.

## 9. 다음 단계 제안 (Ablation 6 반영, 우선순위 순)

1. **관측성 논점 독립 결과로 정리 완료** — [`docs/OBSERVABILITY.md`](OBSERVABILITY.md).
   Ablation 3 + 초기 §4.2(다중 윈도우 병목) + 3D 평면-커버리지(표준 뷰 / `lat_y` gap-closing)는
   순환성과 완전히 무관하게 견고하다. 다중창이 속도/교차빔은 개선하나 WSS는 오히려 낮출 수
   있다는 뉘앙스(Ablation 6)도 포함.
2. **전달형 재설계 구현·사전등록 완료, 근본 제약은 유지** — 파라메트릭 능동수축 forcing을
   세 번째 백본 `fsi_param`(`f = T_max·g(t)·w(x)·d̂(x)`, 자유도 5, 참 `f` 미접근)으로 구현하고
   **Ablation 8(전달형 3-way: A / `fsi_param` / 오라클)**로 사전 등록했다. 결과는 근본 제약을
   재확인한다: 밴드-국소 합성 상한에서 전달형 B는 **Model A와 구별되지 않으며**(speed Δ=0.000,
   pressure Δ=+0.025, 유의하지 않음), 오라클에서만 나타나는 미미한 속도 이득은 전달형이 접근할
   수 없다. 즉 파라메트릭 forcing도 *그 궤적을 재현하지 못하는 한* 이득이 없고, 재현하면
   Ablation 6의 궤적-특이 누출로 돌아간다. 재설계 검증은 상관 게이트(`--which coupling`) +
   **셔플 진단(`--which forcing-shuffle`)**을 모두 통과해야 한다. 실 IBFE forcing조차 깨끗한
   분리자가 아니며 — **밴드-국소**(공동 내부 0 ⇒ 밴드 밖 B=A)이고 u에서 독립적이지 않으므로
   (궤적 정보 3경로) — 실 forcing 실험은 **오라클**(평가 전용 상한)로만 성립한다. 전달물
   Model B(`fsi_param`)의 입력은 **벽 운동학 + 저차원 활성화 파라미터**뿐이고 `f`는 임상
   에코에서 얻을 수 없다.
3. **실제로 푼 FSI 지상진값 확보(핵심 관문)** — 선행 관문은 **해상도 게이트 R**
   (`dx = 0.94 mm`, 짧은 수렴 세그먼트, 클라우드; *아직 미실행*)이다. ("AMR 3레벨"이 아니라
   이 수렴 게이트가 기준.) R 통과 후 심박 1주기 IBFE export → NPZ/매니페스트 →
   `validate_ibfe_frames` → `train_ibfe`로 A/B + forcing 오라클(**support-preserving 셔플** +
   **band-mask** 대조) 재실행. **Minimum Goal B(`dx = 1.875 mm`, 1–2 beat)는 포맷/배관
   검증용일 뿐 과학적 결론 근거가 아니다** — 이 해상도에서 밴드는 공동 체적의 43–59%만 덮으므로
   R이 뚫리기 전까지 압력 관련 결론은 이상적 상한으로만 해석한다. **단 IBAMR 실행 환경 미확보.**
   게이트 R을 기다리지 않고 할 수 있는 우회는 공개 4D-flow / phantom 볼륨
   (`f = 0`)에서 전달형 A vs `fsi_param`을 돌리는 것 — 오라클 실험이 아님.
4. **baseline 노이즈 보정 완료** — 8%/10%는 `placeholder`(기본, Ablation 1–6 재현),
   신규는 `tracking="ste"`(−5%/9%; Houard 2021 CV 8.9%, Farsalinos 2015). `A_exact`는
   `tracking="exact"`. 상세: [`docs/TRACKING_NOISE.md`](TRACKING_NOISE.md).
5. 밸브 데이터가 있으면 `predict_scalar=True`로 잔류시간 복원, 3D는 예산·다중 윈도우 상향.
