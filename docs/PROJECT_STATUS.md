# PINNecho 프로젝트 작업 상태

> 최종 업데이트: 2026-09-20 · 브랜치 `cursor/fsi-pinn-doppler-stage1-935e` · PR #1
> 테스트: **72 passing** (`pytest`)

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
- **순환성과 독립적으로 남는 견고한 결과**: **관측성(다중 음향창)** — FSI 없는 baseline에서도
  교차빔 `u`가 창 수에 단조 반응(0.98→0.75), SNR·희소성은 거의 무관. 단, 다중창이 주 성분 `v`·
  WSS corr는 오히려 낮출 수 있음. 전용 정리: [`docs/OBSERVABILITY.md`](OBSERVABILITY.md).
- **Stage 1 범위**: 실제 환자·에코 데이터 없음. IBAMR/IBFE 파이프라인 출력을 모사한
  **해석적·자기일관(divergence-free, exact no-slip, NS-exact) 합성 지상진값** 위에서 검증.

### 두 백본 (아키텍처는 동일, 물리항만 다름)

| | A — `baseline` | B — `fsi_informed` |
|---|---|---|
| 운동량 forcing `f` | `0` (일반 비압축 NS) | IBAMR/IBFE FSI 체적력 |
| 벽 경계조건 | 운동학적 no-slip (윤곽 추적 속도) | 정확한 FSI 벽속도 + (옵션) traction 연속성 |

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
| A/B 시각화(필드 패널 + 심장주기 GIF) | `eval/visualize.py` | ✅ + 스모크 |
| 관측성 스윕(윈도우/SNR/희소성) | `train/observability.py` | ✅ + 스모크 |
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
| 3D 검증 | `pinnecho-train-3d` | `scripts/train_model_3d.py` |
| 예제 IBFE export 생성 | — | `scripts/make_example_ibfe_export.py` |

공통 옵션: `--dtype {float32,float64}`, `--steps`, `--lbfgs-iters`, `--seed`.

---

## 6. 테스트 현황 (72 passing)

| 파일 | 검증 대상 |
|---|---|
| `test_operators.py` | autograd 미분 연산자 vs 해석해 |
| `test_ns_residual_taylor_green.py` | NS 잔차 ~0 (Taylor–Green, Poiseuille) |
| `test_scalar_transport.py` | 스칼라 전달 잔차 ~0 |
| `test_traction.py` | 유체 traction + 연속성 |
| `test_synthetic_lv.py` / `test_synthetic_lv_3d.py` | 2D/3D 합성 지상진값 자기일관성 |
| `test_doppler.py` / `test_synthesize_doppler.py` | 도플러 투영·잡음·에일리어싱·SNR |
| `test_ibfe_export.py` / `test_ibfe_pipeline.py` | IBFE 인터페이스 + I/O·검증·어댑터 |
| `test_visualize_smoke.py` / `test_observability_smoke.py` | 시각화·스윕 배관 |
| `test_train_model_a_smoke.py` / `test_train_3d_smoke.py` | Model A/B·3D 학습 스모크 |
| `test_ablation_smoke.py` | 물리항 격리·traction 섭동 대조 실험 배관 |
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
- **셔플 진단(실데이터)**: `train_ibfe(..., shuffle_forcing=True)` → Ablation 6 진단을 **실제
  IBFE forcing**에 그대로 재실행. 실 forcing은 u에서 역산한 게 아니라 독립 추정값이므로,
  exact−shuffled 격차가 사라지면 그때는 "물리 사전지식의 순수 효과"로 주장 가능.
- **예제**: `python scripts/make_example_ibfe_export.py --dim {2,3} [--valve]`.

필드 요약: 유체 볼륨(`coords/velocity/pressure/forcing_fluid`), 인터페이스
(`coords/normals/velocity/traction_wall`), 밸브(`coords/velocity_mitral|aortic`),
메타(`cycle_period, rho, mu, spatial_dim`). 단위는 SI, 좌표는 공간→시간 순.

---

## 8. 남은 작업 (데이터 의존)

1. **실제 IBAMR/IBFE export 연결** — 위 계약에 맞춰 `load_ibfe_output(path)`로 투입.
   배관·검증·학습 경로는 모두 구현·테스트 완료.
2. **잔류시간 실측 복원** — 실제 승모판 유입 필드(또는 개방형 공동 합성) 필요.
   스칼라 잔차·`c` 헤드·유입 재초기화·`with_valve` 유입 점군은 완비.
3. **고해상도 3D** — 더 큰 학습 예산 + 다중 윈도우 취득 + 실 3D 지상진값.

## 9. 다음 단계 제안 (Ablation 6 반영, 우선순위 순)

1. **관측성 논점을 독립 결과로 먼저 정리** — Ablation 3 + 초기 §4.2(다중 윈도우 병목)는
   순환성과 완전히 무관하게 견고하다. 다중창이 속도/교차빔은 개선하나 WSS는 오히려 낮출 수
   있다는 뉘앙스(Ablation 6)도 포함해 방법론 결과로 확정.
2. **재설계는 근본 제약을 안고 진행** — 파라메트릭 능동수축 forcing도 *그 궤적을 재현하는 한*
   Ablation 6가 보인 궤적-특이 누출을 피하기 어렵다. 따라서 재설계 검증은 상관 게이트
   (`--which coupling`)뿐 아니라 **셔플 진단(`--which forcing-shuffle`)까지 통과**해야 하며,
   근본적으로는 **독립 추정된 실제 FSI forcing**이 있어야 "물리 사전지식"과 "정답 주입"이
   분리된다.
3. **실제로 푼 FSI 지상진값 확보(핵심 관문, 유일한 순환 제거 경로)** — 선행 관문은
   **cardiac4d-pipeline의 AMR 3레벨 실런 검증**(별도 저장소; 본 PINNecho 워크스페이스 밖) →
   심박 1주기 IBFE export → NPZ/매니페스트로 내보내 `validate_ibfe_frames` → `train_ibfe`로
   A/B + `shuffle_forcing=True` 셔플 진단 재실행. **단 IBAMR 실행 환경(접근/컴퓨팅) 미확보.**
4. **baseline 노이즈 보정** — 8% bias/10% noise를 실제 speckle-tracking reproducibility
   문헌값으로 보정, `A_exact`를 공정 기준선으로 병기.
5. 밸브 데이터가 있으면 `predict_scalar=True`로 잔류시간 복원, 3D는 예산·다중 윈도우 상향.
