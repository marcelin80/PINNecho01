# PINNecho 현재 진행 상황

> 2026-10-12 · 브랜치 `cursor/osu-failcheap-gate-935e` · Corrado 동결, CORe는 윈도우 MATLAB, 팩은 monk
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
| OSU rest 4D-flow (#11) | **윈도우 MATLAB** | monk PATH에는 matlab 없음. 리눅스에 다시 깔 필요 없음. CORe는 윈도우 MATLAB≥2019, `*_core.mat`만 monk로. |
| 실 IBAMR/IBFE 한 주기 | **솔버 export 대기** | gate R `dx = 0.94 mm`. 코라도 제트를 고치지 않음. |

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

## 5. 지금 열린 일 (우선순위)

1. **Corrado는 동결.** A/B/C/블러/D(창을 논문에 안 쓰면) 다시 돌리지 말 것.
2. **지금: 윈도우 MATLAB으로 CORe.** monk venv는 팩/게이트만 한다. 리눅스 MATLAB을 새로 깔지 말 것. k-space는 MATLAB이 있는 쪽에서 받는다. 절차: [`OSU_ZENODO.md`](data/OSU_ZENODO.md).
3. **Charité는 문의만** — 윈도우 CORe를 돌릴 수 없거나 게이트 FAIL 뒤. CFD 금지.
4. **혈전 솔버는 아직 아님.**

Corrado 명령은 [`JET_PRESERVE.md`](results/public_volume/JET_PRESERVE.md)에 기록만. OSU는 [`OSU_ZENODO.md`](data/OSU_ZENODO.md).

### 환자 LV 4D-flow

공개된 **부전·확장 환자 LV 공동 4D-flow 볼륨**은 확인하지 못했다.

- UK Biobank 20213은 참가자당 ~5 MB **2D PC**이지 4D-flow 공동이 아니다. 승인 접근.
- Töger et al. (PLOS ONE 2018) 등 인체 4D-flow 논문은 윤리상 영상을 공유하지 않는다.
- Zenodo [12515230](https://zenodo.org/records/12515230) (OSU, CC BY 4.0): rest **dataset #11** (환자, VENC 200 cm/s). k-space만. MATLAB 없으면 받지 말 것.
- Charité / Vellguth figshare는 private. 논문의 지원자 LV 4D-flow(25 frame, 1.96×1.96×2.8 mm)는 미공개. CFD를 정답으로 쓰지 않음. 문의만.

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
| [`data/OSU_ZENODO.md`](data/OSU_ZENODO.md) | OSU dataset #11 노트북 절차 |
| [`results/public_volume/OSU_GATE.md`](results/public_volume/OSU_GATE.md) | fail-cheap 3항목 (숫자 잠금, MATLAB 대기) |
| [`data/CHARITE_INQUIRY.md`](data/CHARITE_INQUIRY.md) | Charité 문의문 (윈도우 CORe 불가/게이트 FAIL 뒤, CFD 금지) |
