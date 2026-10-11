# PINNecho 현재 진행 상황

> 2026-10-11 · 브랜치 `cursor/public-volume-stasis-935e` · PR #9
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
| PINN vs GT 정체 (1 vs 3 window) | 닫힘 | 창이 제트를 **조금** 지킴. persist Dice는 인용 금지. |
| MRI vs 정합 PIV | 닫힘 | PIV FOV 73 mL는 정체 바다. **30 mL 제트는 밖.** |
| 에코 슬랩 vs 제트 (학습 없음) | 닫힘 | A4C 73%, 다만 공동도 62%(두꺼운 슬랩). |
| 제트 보존 재구성 (`--fast-frac 0.5`) | **코드 준비, 결과 없음** | 균일 샘플이 제트를 안 뽑는 문제. 학습 1회 대기. |
| 라그랑주 잔류시간 / 혈전 솔버 | **하지 않음** | 제트가 새는 필드 위에서는 그림만 나옴. |
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

persist Dice ~0.96은 정체 바다(공동의 84%)다. CHF 동기에 의미 있는 양은 제트다. 창 2개를 더하면 +2.5 mL.

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

## 4. 혈전 / CHF 연동에 대한 현재 답

**지금 이 재구성 필드에 혈전 생성 솔버를 붙이면 안 된다.**

필요한 입력은 “공동이 대체로 느리다”가 아니라, 제트가 어디인지·어디에 오래 머무르는지다. 3-window에서도 제트가 4 mL 새고 speed relL2는 0.42다. persist Dice는 속인다. PIV는 그 제트를 못 본다. 코라도는 강체 펌프라 부전·확장 LV가 아니다.

연동이 가능해지려면 최소:

1. 제트(또는 잔류시간)가 남는 재구성
2. 움직이는 부전 LV — **실 IBFE** 또는 환자 4D-flow
3. 그 다음 기존 정체/혈전 모델

실 IBFE만으로 1이 해결되지 않는다. 코라도의 `f = 0`은 맞는 물리다. IBFE는 벽 운동·압력 GT·오라클 `f`·원래 FSI 질문을 준다. 도플러로 다시 투영하면 같은 관측 문제가 남는다.

---

## 5. 지금 열린 일

**코라도 제트 누수 — 샘플링 레버 (IBFE 불필요).**  
균일 `n_fluid=2000`은 프레임당 제트 ~16점. 3-window 마지막 가중 손실에서 PDE는 ~6%라 지금 내릴 레버가 아니다. `--fast-frac 0.5`가 첫 단일 레버다. 결과는 아직 없다.

```bash
PYTHONPATH=src python3 scripts/public_volume_stasis.py --train \
    --volume data/corrado_lv_phantom/day1_vipr_30min.npz \
    --steps 2000 --n-windows 3 --seed 0 --fast-frac 0.5 \
    --out docs/results/public_volume/stasis_recon_fastfrac.json
```

비교 대상: `jet_ml_pred` / `jet_dice` vs 26.2 mL / ≈0.80.  
`stasis_recon.json`을 덮지 말 것. PDE 가중과 동시에 바꾸지 말 것.

**실 IBFE**는 솔버에서 한 주기를 [`DATA_PREPARATION.md`](DATA_PREPARATION.md) 계약으로 빼는 일이다. PINNecho 로더는 준비되어 있다. gate R(`dx = 0.94 mm`)이 기준이다.

---

## 6. 다시 돌리지 말 것

- 2000 step × 3 seed 관측성
- GT 정체, 1-window / 3-window `--train` (균일 샘플)
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
| [`results/public_volume/STASIS_RECON.md`](results/public_volume/STASIS_RECON.md) | 1 vs 3 window 제트 |
| [`results/public_volume/STASIS_PIV.md`](results/public_volume/STASIS_PIV.md) | MRI vs PIV |
| [`results/public_volume/SLAB_COVERAGE.md`](results/public_volume/SLAB_COVERAGE.md) | 에코 슬랩 |
| [`results/public_volume/JET_PRESERVE.md`](results/public_volume/JET_PRESERVE.md) | `--fast-frac` 실험 |
