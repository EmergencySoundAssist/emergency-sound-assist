# 사이렌 차종 분류 (Subtype) 설계

> **상태: 기본 구성에서 제외** — subtype onnx 를 레포에 싣지 않는다 (성능·속도 이득이 없고 표시 기능만이라 단순화).
> 코드 지원은 유지: `classifier/models/subtype_cnn_attn_s42.onnx` 를 넣으면 자동 활성된다.

> 담당: 소리 분류 모듈 (① 확장)
> 목표: `siren` 으로 판정된 청크 → **구급차 / 경찰차 / 소방차** 세분화 → `ClassResult.subtype` 에 채움
> 최종 출력 예: **"사이렌, 후방, 접근 중" → "구급차, 후방, 접근 중"**

---

## 1. 무엇을 출력하나

검출(`siren / horn / noise`) 위에 한 단계 더 얹는 **2단 분류**다.

```
소리 입력
  ├─ siren? ─→ 차종 추론 ─→ 구급차 / 경찰차 / 소방차 / 긴급차량(불명)
  ├─ horn?  ─────────────→ 경적            (차종 없음)
  └─ noise? ─────────────→ 일반 도로 소음   (차종 없음)
```

- 차종은 **`siren` 일 때만** 채워진다 — `horn`/`noise` 는 `subtype = None`.
- 차종 enum: `SirenSubtype = {AMBULANCE, POLICE, FIRE, UNKNOWN}` ([../interfaces.md](../interfaces.md))
- 최종 종류는 5(+1)가지: 구급차 · 경찰차 · 소방차 · (긴급차량) · 경적 · 일반 도로 소음

---

## 2. 왜 사이렌일 때만 돌리나

차종 모델은 **사이렌 청크로만 학습**됐다 (ViT `subtype_clf.py`, `classes=("siren",)`).
경적·주행음에 넣으면 학습 분포 밖이라 의미 없는 값이 나온다. → 검출이 `siren` 을 줄 때만 게이트.

---

## 3. 모델 — 검출과 같은 몸통, head만 3-클래스

| 항목 | 값 |
|------|-----|
| 출처 | ViT-CNN-Attention `subtype_cnn_attn_s42.pt` → ONNX |
| 구조 | 검출과 **동일한 CNN+Attention 백본** (head 출력만 3) |
| 입력 | 로그멜 `(1, 1, 64, 216)` — **검출과 완전히 동일** |
| 정규화 | 윈도우별 `(x-μ)/σ` — 검출과 동일 |
| 출력 | logits `(1, 3)` = `[구급차, 경찰차, 소방차]` 순서 |

> **핵심 이점**: 입력 멜·정규화가 검출과 같아 **추가 전처리가 0**이다.
> 검출용으로 이미 만든 5초 멜 윈도우를 그대로 차종 모델에 넘긴다.

---

## 4. ONNX 변환 (배포물 만들기)

차종 체크포인트는 `.pt` 라서 ONNX 로 한 번 변환해 `classifier/models/` 에 둔다.
검출과 구조가 같아 ViT 레포의 `export_onnx.py` 를 그대로 쓴다.

```bash
# ViT-CNN-Attention 레포에서 (Python 3.10+)
python export_onnx.py \
  --ckpt models/subtype_cnn_attn_s42.pt --model cnn_attn \
  --out  models/subtype_cnn_attn_s42.onnx

# 산출물을 emergency 쪽으로 복사
cp models/subtype_cnn_attn_s42.onnx  ../emergency-sound-assist/classifier/models/
```

- 변환 후 onnxruntime vs PyTorch 출력 `max|Δ|` 확인 (목표 < 1e-3, 실측 ~1e-6).
- 배치는 고정 1, opset 17 (TensorRT 호환).
- 참고: Python 3.9 환경에서는 ViT 본체(`siren_data.py`)가 최신 타입 문법으로 import 단계에서 막힌다 → `models.py` 만 불러 변환하는 별도 스크립트로 우회.

---

## 5. 추론 흐름 (`classifier/inference.py`)

```
검출 ONNX → siren? ──아니오──→ subtype = None
                  └─예─→ 같은 멜을 차종 ONNX 에 입력
                         → softmax → argmax = j, p = 확률
                         → p ≥ 0.6 : SUBS[j] (구급/경찰/소방)
                           p < 0.6 : UNKNOWN(긴급차량)
```

- **확신 임계 `SUBTYPE_CONF = 0.6`**: 경찰↔구급 혼동이 잦아, 애매하면 `긴급차량` 으로 일반화해 오인을 줄인다.
- 차종 ONNX 세션은 **lazy 로드**(처음 사이렌일 때 한 번).

---

## 6. 정확도 · 한계

- 합성/단일시드 기준 **87~90%** (ViT `subtype_clf.py`).
- **소방차는 잘 분리**(사이클 특성이 뚜렷), **경찰↔구급은 일부 혼동** → `긴급차량` 폴백이 이 약점을 덮는다.
- 차종 ID 는 **비핵심**: 손수 특징으로는 피치×사이클 2D 분리도 ~53%(랜덤 33%) 수준이라, DL로 끌어올린 보조 정보로 본다.
- ⚠ **현재는 합성 데이터 기준** — 실도로 녹음 검증이 다음 관문.

---

## 7. 안전장치 (graceful)

- **차종 ONNX 파일이 없으면** 차종을 조용히 생략하고 기존처럼 `siren`("사이렌")으로만 출력한다 → 검출 파이프라인은 절대 깨지지 않는다.
- 더 강건한 버전: `subtype_cnn_attn_dom_s42.pt`(domain-aug, sim-to-real 강건)를 같은 방식으로 변환해 교체 가능.

---

## 8. 인터페이스 (팀 약속)

- 출력 필드 추가: `ClassResult.subtype: SirenSubtype | None`, `ClassResult.subtype_confidence: float | None`
- 표시 변환: `FusedResult.to_korean()` 이 사이렌+차종이면 차종 이름으로 출력
- 자세한 데이터 약속 → [../interfaces.md](../interfaces.md)
