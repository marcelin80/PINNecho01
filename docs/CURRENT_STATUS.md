# PINNecho 현재 진행 상황

> 2026-10-13 · 브랜치 `cursor/d9-isotropic-v0-935e` · 결정 9 다음 단계: 영상 mesh를 등방 스케일로 V0. 솔버는 5·10 mmHg로 충만.
> 상세 숫자: [`docs/results/public_volume/`](results/public_volume/)
> Stage 1 백본·어블레이션 사전: [`PROJECT_STATUS.md`](PROJECT_STATUS.md)

한 줄: **도플러에서 3D 속도를 재구성하는 방법은 돌아가고, 관측성(창 수)은 실측 LV 볼륨에서도 확인됐다. 그 필드 위에 CHF 혈전 생성 솔버를 지금 얹으면 안 된다.**

---

## 1. 이 프로젝트가 하려는 것

입력은 희소·잡음·**빔 방향 한 성분**뿐인 도플러 유사 속도다. 출력은 좌심실 안 속도장(그리고 원래는 압력·와도·잔류시간)이다.

원래 핵심 질문(FSI 정보로 압력·구배가 좋아지는가)은 **기각**됐다. 순환(`f ≈ ∇p`)과 관측 부족의 하류 증상이었다. 남는 견고한 결과는 **관측성**: 창을 늘리면 안 보이는 성분이 돌아온다. 전달형 Model B(`fsi_param`)는 참 `f` 없이 baseline과 구별되지 않는다. `fsi_informed`는 임상 입력이 아니라 오라클 상한이다.

CHF 혈전 아이디어는 그 다음 층이다: 재구성한 3D 속도 → 정체/잔류시간 → (나중에) 혈전 모델. 속도만으로는 혈전이 아니다.

---

## 2. 지금 어디에 있는가

| 층 | 상태 | 한 줄 |
|---|---|---|
| Stage 1 합성 PINN (A / 오라클 B / 전달 B) | 닫힘 | 압력·오라클 이득 기각. 관측성만 남음. |
| IBFE I/O·검증·어댑터 | 코드 완료 | 디스크 계약만 맞으면 로드. **실 export는 대기.** |
| 공개 4D-flow 계약 (`f = 0`, `p = 0`) | 닫힘 | IBFE가 아님. 질문은 관측성. |
| Corrado LV 팬텀 변환 | 닫힘 | Day 1 PC VIPR 30 min → NPZ. 다시 변환하지 말 것. |
| 창 1/2/3 관측성 (2000 step, 3 seed) | 닫힘 | `u` 1→2, `v` 1→3. speed 0.51 → 0.42. |
| Eulerian 정체 GT (10 cm/s) | 닫힘 | 195 mL 공동, persist 165 mL, **제트 30.4 mL**. |
| PINN vs GT 정체 (3-window, 3 seed) | **A 기록** | 10 cm/s 제트 24.2–26.2 mL. 15 cm/s Dice ~0.56. |
| 1창 3시드 (C) | **기록** | 10 cm/s 23.4–24.0 mL. 창 효과는 작다. 15 cm/s는 실패. |
| MRI vs 정합 PIV | 닫힘 | PIV FOV 73 mL는 정체 바다. **30 mL 제트는 밖.** |
| 에코 슬랩 vs 제트 (학습 없음) | 닫힘 | A4C 73%, 다만 공동도 62%(두꺼운 슬랩). |
| 제트 보존 (`|v_beam|` oversample) | **B 기각** | 15 cm/s −23/−32/−26 %. 피크비 ~0.83. 샘플링 레버 접음. |
| 블러 GT 상한 (σ=1, 학습 없음) | **기록** | 15 cm/s Dice 0.760, 10 cm/s 0.911. PINN은 둘 다 아래. |
| 라그랑주 잔류시간 / 혈전 솔버 | **하지 않음** | 제트가 새는 필드 위에서는 그림만 나옴. |
| OSU rest 4D-flow (#11) | **팩 PENDING, 게이트 1 개정** | (20,77,96,72). 폐기: LV p99≥10 cm/s (잡음만으로 통과). 대체: 조직 3× time-RMS, 푸리에 1–4 ≥0.55, Aao±40%, 승모/대동맥 SV 균형, jet_chamber. 자동박스 43:60 59:69 15:37은 LV 아님. |
| DUPLEX (CREATIS 합성 듀플렉스) | **PINN GT 불가** | 무학습 점검 닫힘. 공개 `.mat`에 `(u,w)` 없음. 투영/발산 계산 불가. [`DUPLEX.md`](data/DUPLEX.md). |
| 열화 대비책 (Corrado+합성기) | **기준만 잠금, 미실행** | 우리 모델이라고 명시. 임상 Nyquist 1.078는 p99 0.261이라 비활성 가능. [`DEGRADE_CORRADO.md`](results/public_volume/DEGRADE_CORRADO.md). |
| 논문 초안 | **접촉 없는 구조** | 지표 결함 + 순환 + (선택) 우리 열화. 환자 정확도/FSI 효과 금지. [`PAPER.md`](PAPER.md). |
| 실 IBAMR/IBFE 한 주기 | **솔버 export 대기** | gate R `dx = 0.94 mm`. 코라도 제트를 고치지 않음. |
| 결정 9 무응력 기준 | **a1 적용됨, 충만 진행** | monk `LV_V0.e` α=0.819292. `scale_fill`이 8 mmHg로 부풀리는 중. 확장기 가설은 아직 불가. [`DECISION_9_APPROX.md`](DECISION_9_APPROX.md). |

---

## 3. Corrado에서 확정된 숫자

데이터: Wisconsin in-vitro LV (Corrado et al., *Magn Reson Med* 2021; OSF 2naw4). 강체 실리콘 + 펌프. 20 프레임, 0.75 s, 2 mm, `f = 0`.

### 관측성 (baseline, 2000 Adam + 50 L-BFGS, 3 seed)

| 창 | speed | u | v | w |
|---|---|---|---|---|
| 1 | 0.506 | 0.728 | 0.898 | **0.260** |
| 2 | 0.432 | **0.409** | 0.834 | 0.339 |
| 3 | **0.418** | 0.423 | **0.690** | 0.356 |

제조 데이터가 아닌 LV 공동에서 창 레버가 재현됐다. 필드를 “복원했다”는 뜻이 아니다.

### 정체 재구성 (10 cm/s, seed 0)

| | GT | 1 window | 3 windows |
|---|---|---|---|
| persist | 164.6 mL | 171.3 | 168.8 |
| **제트** | **30.4 mL** | **23.7 (−22%)** | **26.2 (−14%)** |
| persist Dice | | 0.959 | 0.967 |
| **jet Dice** | | **0.748** | **≈ 0.80** |
| speed relL2 | | 0.516 | 0.417 |

persist Dice ~0.96은 정체 바다(공동의 84%)다. 제트 mL은 컷이 만든 나머지다.

3-window × 3 seed ([`STASIS_RECON_SEEDS.md`](results/public_volume/STASIS_RECON_SEEDS.md)):

| cut | GT jet | pred mean (range) | Dice mean |
|---|---|---|---|
| 5 cm/s | 79.7 mL | 73.2 (72.0–74.5) | 0.84 |
| 10 cm/s | 30.4 mL | **25.3 (24.2–26.2)** | **0.81** |
| 15 cm/s | 10.8 mL | **6.3 (5.3–8.0)** | **0.56** |

10 cm/s 누수 4–6 mL는 시드에 걸쳐 있다. 15 cm/s 코어가 실제로 실패한다. 5 cm/s Dice는 큰 나머지라 높아 보인다.

C 1창 3시드 ([`STASIS_RECON_W1.md`](results/public_volume/STASIS_RECON_W1.md)): 10 cm/s **23.7 mL** (−22/−21/−23 %) vs A 25.3 mL. 창당 점이 더 많은데도 1창이 세 시드 모두 나쁘다(시드 1은 거의 동률). 15 cm/s는 시드 2에서 1창이 더 낫다. 창은 10 cm/s 나머지를 조금 지키고, 코어는 살리지 못한다. `n_data` 고정 교란은 남는다. D는 창을 논문 주장으로 쓸 때만.
`w` relL2는 창이 늘면 나빠진다(0.260→0.356). 희석인지 가리는 것도 D다.

B (`|v_beam|` 프레임별 p80, [`STASIS_RECON_FASTFRAC.md`](results/public_volume/STASIS_RECON_FASTFRAC.md)): 미리 적은 기준으로 **기각**. 15 cm/s 부호 오차 −23.3 / −31.9 / −26.1 %. 시드 1·2가 −25%를 넘는다. 10 cm/s는 28.6 mL(정밀도 0.84–0.87)로 가짜 제트가 아니고, 느린 영역 속도도 나빠지지 않았다. GT 제트 피크비 중앙값 **~0.80–0.88**. 샘플링은 접는다. 발견은 고속 코어의 체계적 과소추정이다.

블러 GT (σ=1복셀, [`STASIS_BLUR_GT.md`](results/public_volume/STASIS_BLUR_GT.md)): 15 cm/s Dice **0.760** (0.6도 0.9도 아님). 10 cm/s는 **0.911**. A/B는 두 상한 아래다. 블러 피크비 0.88–0.90이라 진폭 편향의 상당 부분은 1복셀 부드러움과 맞다. PINN은 그보다 위치를 더 틀린다 (15 cm/s 정밀도 블러 0.98 vs B 0.74–0.85).

### PIV

같은 격자, `origin_delta = 0`. PIV 마스크 73 mL는 MRI 공동의 37%이고 **전부 정체 바다**다. 겹침에서 MRI 제트 0.0 mL, PIV 제트 0.016 mL. PIV는 PINN 제트를 검증할 수 없다. 자동컷 2.4 cm/s의 “제트 55 mL”는 인용 금지.

### 에코 슬랩 (학습 없음, AABB 뷰, 두께 0.12×extent)

| protocol | 제트 | frac | 공동 슬랩 |
|---|---|---|---|
| a4c | 22.3 / 30.4 | 0.734 | 122 / 195 (0.62) |
| a4c+a2c | 24.6 | 0.809 | 160 |
| a4c+plax | 22.3 | 0.734 | 122 (A4C와 동일 면) |
| 4-view | 24.7 | 0.813 | 170 |

제트가 “4D-flow 전용”은 아니다. 슬랩이 두꺼워 공동의 62%도 A4C에 들어간다. 농축은 73% vs 62%로 약하다. 임상 5 mm 슬랩은 아직 안 돌렸다.

---

## 4. 논문 프레임 (정직하게)

관측성만으로는 새롭지 않다. triplane iVFM이 이미 다중 평면으로 3성분을 복원하고, 단일 빔이 한 성분만 보는 것은 기하다. 이 프로젝트가 방어하기 쉬운 중심은:

- relL2와 persist Dice가 **임상적으로 중요한 제트 손실을 가린다**
- 재구성은 **하류 과제(정체, 나중의 혈전) 기준**으로 평가해야 한다
- 제조해·오라클 `f`를 입력으로 쓸 때의 **순환 목록**

관측성은 그 안의 한 결과로 넣는다. “창을 늘리면 좋아진다”를 제목으로 두지 않는다.

## 4b. 혈전 / CHF 연동에 대한 현재 답

**지금 이 재구성 필드에 혈전 생성 솔버를 붙이면 안 된다.**

필요한 입력은 “공동이 대체로 느리다”가 아니라, 제트가 어디인지·어디에 오래 머무르는지다. B 이후에도 15 cm/s 코어는 23–32% 새고 피크비는 ~0.83이다. persist Dice는 속인다. PIV는 그 제트를 못 본다. 코라도는 강체 펌프라 부전·확장 LV가 아니다.

연동이 가능해지려면 최소:

1. 제트(또는 잔류시간)가 남는 재구성
2. 움직이는 부전 LV — **실 IBFE** 또는 환자 4D-flow
3. 그 다음 기존 정체/혈전 모델

실 IBFE만으로 1이 해결되지 않는다. 코라도의 `f = 0`은 맞는 물리다. IBFE는 벽 운동·압력 GT·오라클 `f`·원래 FSI 질문을 준다. 도플러로 다시 투영하면 같은 관측 문제가 남는다.

---

## 4c. 결정 9 — 지금 확장기 가설 쪽에 알릴 것

**결정 9가 구현되기 전의 박동 데이터는 수축과 박출 구간만 쓸 수 있다. 확장기 충만 구간은 비생리적이다.**

PINNecho의 핵심 관심사(E파, IVPG, vortex ring)는 심실이 실제로 수십 mL를 채워야 존재한다. 영상 부피를 무응력 기준으로 두면 충만이 거의 없어 E파가 나오지 않는다. 그래서 (b) 영상=기준, (c) 한계로 기록은 목적을 포기하는 선택이다. 잠긴 경로는 **(a)**: 영상을 이미 부하가 걸린 상태로 보고 무응력 V0를 만든다.

- **근사(먼저)**: 영상 mesh를 등방 스케일로 줄인다. `(V0/V_img)^{1/3}`. Klotz V0 ≈ 58 mL는 단일 박동 경험식이라 오차가 크고, pat12는 선천성 심질환이라 순응도가 다를 수 있다. `p_img`는 모른다 — mid~late diastole 5–10 mmHg를 가정하고 두 값에서 민감도를 본다. 벽 두께가 같이 줄어 ~7.4 → ~6 mm가 되고 변형 상태도 실제와 다르다. 문서에 근사라고 쓴다.
- **정식(나중에)**: `p_img` 아래서 영상 형상이 되는 무응력 형상의 역변환. IBFE 반복마다 정적 해석. 이 저장소에서 돌리지 않는다.
- **attempt 2는 돌려도 된다.** V0가 그대로이므로 충만·EF는 풀리지 않는다. 예측은 Vmax 108–110 mL, 순서만. 목적은 판막 이벤트 순서, 박출 끝 ESV, AV 닫힘. ESV 절댓값은 `T_max`·후부하·IVC 누설에 의존한다. 재튜닝 2회째이고, 3회째는 결정 9 이후.
- **`q_close`**: 열린 동안 `p_lv = p_la`라 압력 기반 전환이 안 된다. 그래서 Q로 우회한다.

표시 위치: `/meta/fidelity` (`diastole_usable=false`가 기본). 검증기가 확장기 경고를 낸다. [`DECISION_9.md`](DECISION_9.md).

---

## 5. 지금 열린 일 (우선순위)

1. **결정 9 a1는 monk에서 끝.** `LV_V0.e`, α=0.819292, V0=58.000 mL, V_img=105.466 mL. PINNecho로 다시 줄이지 말 것. **지금 할 일:** `~/lv_runs/scale_fill`(8 mmHg)이 끝날 때까지 두고 V가 ~105 mL 근처인지, `max |J-1|`이 0.8 아래에서 멈추는지 본다. 그 전에는 확장기 가설에 쓰지 말 것. [`DECISION_9_APPROX.md`](DECISION_9_APPROX.md).
2. **논문 초안을 접촉 없이 쓴다.** [`PAPER.md`](PAPER.md). Charité / Delft / OSU 마스크 / 협업 문의는 필수 경로에서 뺀다.
3. **Corrado A/B/C/블러는 동결.** 피크비 ~0.83은 기록된 B 표. 재실행하지 말 것.
4. **DUPLEX로 학습하지 말 것.** 공개본은 벡터 GT가 없다. iVFM 논문 숫자와 같은 데이터인지는 미확인 → 같은 표에 넣지 말 것.
5. **열화 문단이 필요할 때만** Corrado+우리 합성기. `ideal` 먼저, 하락폭만. `alias_clinical` wrap < 0.01이면 행을 뺀다.
6. **MIMIC은 선택.** 정답 없음, credentialing에 reference 필요.
7. **혈전 솔버는 아직 아님.**

Corrado 명령은 [`JET_PRESERVE.md`](results/public_volume/JET_PRESERVE.md)에 기록만. OSU는 [`OSU_ZENODO.md`](data/OSU_ZENODO.md).

### 환자 LV 4D-flow

공개된 **부전·확장 환자 LV 공동 4D-flow 볼륨**은 확인하지 못했다.

- UK Biobank 20213은 참가자당 ~5 MB **2D PC**이지 4D-flow 공동이 아니다. 승인 접근.
- Töger et al. (PLOS ONE 2018) 등 인체 4D-flow 논문은 윤리상 영상을 공유하지 않는다.
- Zenodo [12515230](https://zenodo.org/records/12515230) (OSU, CC BY 4.0): rest **dataset #11** (환자, VENC 200 cm/s). k-space만. MATLAB 없으면 받지 말 것.
- Charité / Vellguth figshare는 private. 논문의 지원자 LV 4D-flow(25 frame, 1.96×1.96×2.8 mm)는 미공개. CFD를 정답으로 쓰지 않음. 문의만.
- DUPLEX (CREATIS): 합성 A3C 색도플러. 볼륨이 아님. 공개 mat에 CFD 벡터 없음. [`DUPLEX.md`](data/DUPLEX.md).

둘 다 피험자 1명 수준이라 사례 시연이다. 부전·확장 코호트는 여전히 없다.

---

## 6. 다시 돌리지 말 것

- 2000 step × 3 seed 관측성
- GT 정체, 1-window / 3-window `--train` (균일 샘플)
- B `--fast-frac 0.5` 3-seed
- C `--n-windows 1` 3-seed
- `--blur-sigma 1.0`
- PIV 변환·겹침 비교
- `thickness_frac = 0.12` 슬랩
- `fsi_informed` on 공개 볼륨
- 이 필드 위 Lagrangian / 혈전 솔버

노트북 JSON은 `docs/results/public_volume/`에 그대로 둔다.

---

## 7. 문서 지도

| 문서 | 내용 |
|---|---|
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | Stage 1 컴포넌트·어블레이션 (2026-09 백본) |
| [`PUBLIC_VOLUME.md`](PUBLIC_VOLUME.md) | 공개 볼륨 계약 |
| [`OBSERVABILITY.md`](OBSERVABILITY.md) | 제조 데이터 관측성 |
| [`DATA_PREPARATION.md`](DATA_PREPARATION.md) | 실 IBFE export 계약 |
| [`results/public_volume/CORRADO_OBSERVABILITY.md`](results/public_volume/CORRADO_OBSERVABILITY.md) | 코라도 창 스윕 |
| [`results/public_volume/STASIS_GT.md`](results/public_volume/STASIS_GT.md) | 10 cm/s GT |
| [`results/public_volume/STASIS_RECON.md`](results/public_volume/STASIS_RECON.md) | 1 vs 3 window (seed 0) |
| [`results/public_volume/STASIS_RECON_SEEDS.md`](results/public_volume/STASIS_RECON_SEEDS.md) | 3-window × 3 seed × 3 cut (A) |
| [`results/public_volume/STASIS_RECON_FASTFRAC.md`](results/public_volume/STASIS_RECON_FASTFRAC.md) | B `|v_beam|` p80, 기각 |
| [`results/public_volume/STASIS_RECON_W1.md`](results/public_volume/STASIS_RECON_W1.md) | C 1창 3시드 |
| [`results/public_volume/STASIS_BLUR_GT.md`](results/public_volume/STASIS_BLUR_GT.md) | σ=1 블러 GT 상한 |
| [`results/public_volume/STASIS_PIV.md`](results/public_volume/STASIS_PIV.md) | MRI vs PIV |
| [`results/public_volume/SLAB_COVERAGE.md`](results/public_volume/SLAB_COVERAGE.md) | 에코 슬랩 |
| [`results/public_volume/JET_PRESERVE.md`](results/public_volume/JET_PRESERVE.md) | `|v_beam|` 실험 |
| [`data/OSU_ZENODO.md`](data/OSU_ZENODO.md) | OSU #11: 윈도우 CORe → monk 팩 |
| [`results/public_volume/OSU_GATE.md`](results/public_volume/OSU_GATE.md) | fail-cheap 5항목 (p99 폐기, 조직·푸리에·판막 균형 잠금) |
| [`data/CHARITE_INQUIRY.md`](data/CHARITE_INQUIRY.md) | Charité 문의문 (윈도우 CORe 불가/게이트 FAIL 뒤, CFD 금지) |
| [`data/DUPLEX.md`](data/DUPLEX.md) | CREATIS 합성 듀플렉스: 무학습 점검 닫힘, PINN GT 불가 |
| [`results/public_volume/DEGRADE_CORRADO.md`](results/public_volume/DEGRADE_CORRADO.md) | Corrado+우리 합성기 열화 사다리 (미실행) |
| [`PAPER.md`](PAPER.md) | 접촉 없는 논문 초안 |
| [`DECISION_9.md`](DECISION_9.md) | 무응력 기준 (a) 잠금. `/meta/fidelity`. 결정 9 이전 박동은 수축·박출만. |
| [`DECISION_9_APPROX.md`](DECISION_9_APPROX.md) | 등방 스케일 절차. 솔버 충만 5·10 mmHg. |
