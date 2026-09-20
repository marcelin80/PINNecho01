# PINNecho 프로젝트 작업 상태

> 최종 업데이트: 2026-09-20 · 브랜치 `cursor/fsi-pinn-doppler-stage1-935e` · PR #1
> 테스트: **58 passing** (`pytest`)

FSI 정보 기반 물리정보신경망(PINN)으로 희소·단일성분 도플러 측정에서 좌심실 내부
유동장(속도·압력·와도·잔류시간)을 복원하고, 두 물리 백본을 비교하는 프로젝트의
현재 상태를 정리한 문서입니다.

---

## 1. 목표와 핵심 질문

- **입력**: 희소하고 잡음이 있으며 **빔 방향 성분만** 측정되는 도플러 유사 속도 데이터.
- **출력**: 전체 속도장, 압력, 와도, 벽 전단응력(WSS), (실데이터 확보 시) 잔류시간.
- **핵심 과학 질문**: Navier–Stokes 백본에 심장 **유체-구조 상호작용(FSI)** 정보를
  더하면, 운동학적 no-slip 기반 기존 방법(AI-VFM, iVFM-PINN, CSF-PINN) 대비
  **속도 구배량(와도·WSS)과 압력** 복원이 개선되는가? → **개선됨 (아래 결과 참조).**
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

### 4.1 Model A vs Model B (벽속도 분기, 동일 아키텍처/데이터/초기화)

상대 L2 오차(↓), 괄호는 상관계수(↑):

| 지표 | A | B (forcing+FSI 벽) | B (+traction) |
|---|---|---|---|
| speed | 0.481 | 0.454 | 0.452 |
| vorticity | 0.585 (0.81) | 0.561 (0.83) | **0.549 (0.84)** |
| **WSS** | 0.618 (0.61) | 0.489 (0.69) | **0.487 (0.68)** |
| **pressure** | 0.971 (0.24) | 2.209 (0.79) | **0.105 (0.996)** |

- 현실적인 운동학적 벽 오차를 주면 **WSS 개선폭이 ~21%**로 확대(벽 구배량이라 정확한
  FSI 벽속도의 이득이 큼).
- **traction 연속성**을 켜면 압력이 거의 정확히 복원(상관 0.24 → 0.996).
- 원자료: `docs/results/stage2_forcing.json`, `docs/results/stage2_traction.json`.

### 4.2 관측성 스윕 (Model B, 1800 Adam + 200 L-BFGS, float32)

| 조건 | SNR(dB) | 윈도우 | pts/frame | speed | `u` | `v` | WSS | pressure |
|---|---|---|---|---|---|---|---|---|
| windows=1 | 26 | 1 | 300 | 0.568 | 0.969 | 0.564 | 0.741 | 0.219 |
| **windows=2** | 26 | 2 | 300 | **0.555** | **0.815** | 0.579 | **0.725** | **0.172** |
| noise=0.02 | 34 | 1 | 300 | 0.569 | 0.972 | 0.563 | 0.744 | 0.221 |
| noise=0.10 | 20 | 1 | 300 | 0.574 | 0.968 | 0.582 | 0.756 | 0.228 |
| points=150 | 26 | 1 | 150 | 0.573 | 0.972 | 0.545 | 0.755 | 0.232 |
| points=600 | 26 | 1 | 600 | 0.562 | 0.972 | 0.564 | 0.743 | 0.229 |

- **지배적 변수는 각도(윈도우 수)**. 심첨 단일 윈도우는 `v`는 잘 보지만 교차빔 `u`는
  거의 못 봄(0.97). 두 번째 각도 윈도우가 `u`를 **0.82로 급감**시키고 speed·WSS·압력도 개선.
- SNR(20–34dB)·희소성(150–600)은 ~1–2%만 변동 → 병목은 **관측 방향 수**.
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

## 6. 테스트 현황 (58 passing)

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

## 9. 다음 단계 제안

- 실 IBFE 프레임 1주기를 NPZ/매니페스트로 내보내 `validate_ibfe_frames`로 계약 확인.
- 동일 frames로 `train_ibfe`를 baseline·fsi_informed 두 번 실행해 실데이터 A/B 확보.
- 밸브 데이터가 있으면 `predict_scalar=True`로 잔류시간 복원 착수.
- 3D는 예산 상향(더 큰 네트워크·스텝, GPU/float32)과 3+윈도우 취득으로 정확도 개선.
