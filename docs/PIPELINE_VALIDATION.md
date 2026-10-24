# PINNecho verdict — pipeline only (2026-10-24)

Locked answer to the solver-side question:

> 확장기 충만·판막 순서는 통과했다. 후반 충만 정체(111→27 mL/s)는
> 심낭 대시팟 η=0.05였고, η=0에서 D2≈130 mL/s로 첫 게이트와 같다.
> 첫 박동은 예열(ED 87.7 mL은 짧은 확장기), 다음 ED는 90–92 mL로 본다.
> EF는 33%(V0=58 mL), dx=1.875 mm, /meta/fidelity=pipeline-validation.
> 이 데이터로 관계·파이프라인 확인이면 충분한가? 생리 E파/IVPG 평가면
> 부족한가?

## Answer

**파이프라인만 — 지금 이 박동에 대해서.** 관계(배관: 판막 순서, 충만
존재, export 계약)는 이 시리즈로 확인할 수 있다.

생리 E파 / IVPG / vortex ring은 PINNecho에 **필요하다.** 그것이
프로젝트의 핵심 관심사다. “부족”은 목표가 아니라는 뜻이 아니다.
**이 dx=1.875 mm 예열 박동으로는 그 가설을 평가할 수 없다**는 뜻이다.
목적지는 그대로다. 지금 export가 그 평가를 열지 못할 뿐이다.

`/meta/fidelity` = `pipeline-validation`
(`purpose`, or a raw `fidelity: pipeline-validation` key).
`diastole_usable` stays **false**. Filling may be present
(`filling_present`, usable phase `filling_pipeline`).
`e_wave_ivpg_usable` is false.

## Why pipeline is enough

- Valve order passed.
- Fill exists. The late stall 111→27 mL/s is a diagnosed dashpot
  (η=0.05), not “no fill”. η=0 restores D2≈130 mL/s, same as the
  first gate.
- First-beat ED 87.7 mL is warmup / short diastole. Projected next
  ED 90–92 mL is a pipeline close-out number, not an image match.
- `dx=1.875 mm` is **Minimum Goal B**: format and plumbing. That is
  exactly this tag.

## Why *this beat* cannot evaluate E-wave / IVPG

The project still needs those quantities. This series cannot support
the claim.

- Image cavity is ~105 mL. Warmup ED 87.7 and next 90–92 are still
  short. ESV at EF 33% and V0=58 sits near V0 (~59 mL).
- Decision 9 a1 is an approximation (wall 7.4→6.06 mm; strain state
  is not an unload).
- IVPG and E-wave need gate **R** (`dx=0.94 mm`). Band coverage at
  1.875 mm is 43–59% of the cavity. Pressure-gradient claims are
  not allowed here.
- **B4 (V0/EF) at this dx does not unlock E-wave / IVPG.** It only
  changes how much the cavity can fill. Spend tens of hours on B4
  after the pipeline is closed, not instead of closing it, and not
  to rescue a physiological claim on Minimum Goal B.

## Next

η=0으로 박동 2–3을 닫는다 (~11 h, 이어서 한 박동 더). 그 런도
`pipeline-validation`이다. 그 다음이 B4(V0/EF, 충만량)이고, E파/IVPG
평가는 **gate R (`dx=0.94 mm`)** 이후의 그 목표 데이터에서 한다.

Code: `pipeline_validation_fidelity()` in
`pinnecho.data.ibfe_fidelity`.
