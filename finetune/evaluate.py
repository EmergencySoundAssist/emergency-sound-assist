"""baseline 측정 하네스 — manifest 를 읽어 CER·키워드 히트율을 조건별로 리포트.

- 엔진: faster-whisper 를 직접 호출하되 파라미터는 배포 설정(STTConfig 기본값)을
  미러링(language=ko, beam=1, condition_on_previous_text=False, 환각 가드 3종).
  발화 단위 파일이므로 VAD·버퍼링은 미경유(배포 경로와의 차이 — 리포트에 명기).
- --bias: 데이터-0 대안 arm. initial_prompt + hotwords 를 켠다(런타임 stt/ 는 불변).
- CER 은 jiwer corpus-level(전체 편집거리 합 / 참조 길이 합), 정규화는 metrics 참고.

실행 예:
  python3 -m finetune.evaluate --model small --limit 10   # 스모크
  python3 -m finetune.evaluate --model medium --bias
  python3 -m finetune.evaluate --combine                  # arm 비교표
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import jiwer

from .audio_io import load_mono_16k
from .metrics import keyword_hit, normalize_ko
from .phrases import load_phrases

REPORT_DIR = Path("data/eval_v0/reports")
MANIFEST = Path("data/eval_v0/manifest.jsonl")
PHRASES_CSV = Path("finetune/emergency_phrases.csv")


# ---------------------------------------------------------------------------
# 집계 (순수 — 테스트 대상)
# ---------------------------------------------------------------------------
def _group_metrics(pairs: list[tuple[str, str]]) -> dict:
    refs = [normalize_ko(r) for r, _ in pairs]
    hyps = [normalize_ko(h) for _, h in pairs]
    keep = [(r, h) for r, h in zip(refs, hyps) if r]     # 빈 참조 방어
    if not keep:
        return {"cer": None, "n": 0}
    refs = [r for r, _ in keep]
    hyps = [h for _, h in keep]
    return {"cer": float(jiwer.cer(refs, hyps)), "n": len(keep)}


def _cond_key(r: dict) -> str:
    return r["condition"] if r["snr_db"] is None else f"{r['condition']}@{r['snr_db']}"


def _src_group(r: dict) -> str:
    """게이트 판단용 소스 그룹: TTS(합성) vs corpus(실음성)."""
    return "tts" if r["source"] == "tts" else "corpus"


def aggregate(rows: list[dict]) -> dict:
    all_pairs = [(r["ref"], r["hyp"]) for r in rows]
    by_cond: dict[str, list] = defaultdict(list)
    by_src: dict[str, list] = defaultdict(list)
    by_cs: dict[str, list] = defaultdict(list)           # 소스그룹/조건 교차
    kw_by_cond: dict[str, list] = defaultdict(list)
    for r in rows:
        key = _cond_key(r)
        by_cond[key].append((r["ref"], r["hyp"]))
        by_src[r["source"]].append((r["ref"], r["hyp"]))
        by_cs[f"{_src_group(r)}/{key}"].append((r["ref"], r["hyp"]))
        if r.get("keywords"):
            kw_by_cond[key].append(keyword_hit(r["keywords"], r["hyp"]))

    hits = [h for hs in kw_by_cond.values() for h in hs]

    overall = _group_metrics(all_pairs)
    # 참고용 WER(정규화 없이 원문 그대로 — 띄어쓰기 차이도 오류로 센다)
    refs_w = [r["ref"] for r in rows if r["ref"].strip()]
    hyps_w = [r["hyp"] for r in rows if r["ref"].strip()]
    overall["wer"] = float(jiwer.wer(refs_w, hyps_w)) if refs_w else None

    return {
        "overall": overall,
        "by_condition": {k: _group_metrics(v) for k, v in sorted(by_cond.items())},
        "by_source": {k: _group_metrics(v) for k, v in sorted(by_src.items())},
        "by_condition_source": {k: _group_metrics(v) for k, v in sorted(by_cs.items())},
        "keyword_hit_rate": (sum(hits) / len(hits)) if hits else None,
        "keyword_n": len(hits),
        "keyword_hit_by_condition": {k: sum(v) / len(v)
                                     for k, v in sorted(kw_by_cond.items())},
    }


# ---------------------------------------------------------------------------
# 편향 arm (데이터-0 대안)
# ---------------------------------------------------------------------------
def make_bias_kwargs(phrases_rows: list[dict]) -> dict:
    seen, kws = set(), []
    for r in phrases_rows:
        for k in r["keywords"]:
            if k not in seen:
                seen.add(k)
                kws.append(k)
    return {
        "initial_prompt": "도로에서 들리는 한국어 안내와 경고. "
                          "차 세우세요. 구급차 지나갑니다. 길 터주세요. 조심하세요.",
        # 고유 키워드는 ~72개(≈300자) — 전 카테고리가 들어가도록 여유 상한 80
        "hotwords": " ".join(kws[:80]),
    }


# ---------------------------------------------------------------------------
# 실행 (transcribe_fn 주입 가능 — 테스트는 가짜 함수 사용)
# ---------------------------------------------------------------------------
def _make_whisper_fn(model_size: str, bias: bool):
    from faster_whisper import WhisperModel

    from stt.config import STTConfig
    from stt.device import resolve_runtime

    cfg = STTConfig(model_size=model_size)
    device, compute_type = resolve_runtime(cfg.device, cfg.compute_type)
    print(f"[eval] 모델 로드: {model_size} on {device}/{compute_type}", file=sys.stderr)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    extra = make_bias_kwargs(load_phrases(PHRASES_CSV)) if bias else {}

    def fn(path: str) -> str:
        segments, _info = model.transcribe(
            load_mono_16k(path), language=cfg.language, beam_size=cfg.beam_size,
            condition_on_previous_text=False,
            no_speech_threshold=cfg.no_speech_threshold,
            log_prob_threshold=cfg.log_prob_threshold,
            compression_ratio_threshold=cfg.compression_ratio_threshold,
            **extra,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    return fn


def run_eval(manifest_path, out_dir, arm_name: str, transcribe_fn,
             limit: int | None = None) -> dict:
    with open(manifest_path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if limit:
        items = items[:limit]

    rows, t0 = [], time.time()
    for n, it in enumerate(items, 1):
        hyp = transcribe_fn(it["path"])
        rows.append({"ref": it["ref_text"], "hyp": hyp, "condition": it["condition"],
                     "snr_db": it["snr_db"], "source": it["source"],
                     "keywords": it.get("keywords")})
        if n % 25 == 0:
            print(f"[eval] {arm_name}: {n}/{len(items)} ({time.time()-t0:.0f}s)",
                  file=sys.stderr)

    report = aggregate(rows)
    report["arm"] = arm_name
    report["elapsed_sec"] = round(time.time() - t0, 1)
    report["items"] = rows                      # 오답 분석용 원본 페어 포함

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{arm_name}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    _write_md(out_dir / f"{arm_name}.md", report)
    return report


def _fmt(v) -> str:
    return "-" if v is None else f"{100*v:.1f}%"


def _write_md(path: Path, rep: dict) -> None:
    lines = [f"# eval report — {rep['arm']}", "",
             f"- n={rep['overall']['n']}, 소요 {rep.get('elapsed_sec','?')}s",
             f"- **CER(공백불문) {_fmt(rep['overall']['cer'])}**, "
             f"WER(참고) {_fmt(rep['overall'].get('wer'))}",
             f"- 긴급문구 키워드 히트율: {_fmt(rep['keyword_hit_rate'])} "
             f"(n={rep['keyword_n']})", "",
             "| 그룹 | CER | n |", "|---|---|---|"]
    for k, m in rep["by_condition"].items():
        lines.append(f"| 조건 {k} | {_fmt(m['cer'])} | {m['n']} |")
    for k, m in rep["by_source"].items():
        lines.append(f"| 소스 {k} | {_fmt(m['cer'])} | {m['n']} |")
    for k, m in rep["by_condition_source"].items():
        lines.append(f"| 교차 {k} | {_fmt(m['cer'])} | {m['n']} |")
    lines.append("")
    lines.append("| 조건(긴급문구) | 키워드 히트율 |")
    lines.append("|---|---|")
    for k, v in rep["keyword_hit_by_condition"].items():
        lines.append(f"| {k} | {_fmt(v)} |")
    lines.append("")
    lines.append("주의: 발화 단위 파일 평가(VAD/버퍼링 미경유). TTS 소스 절대치는 "
                 "낙관/비관 편향 가능 — arm 간 상대 비교용.")
    path.write_text("\n".join(lines), encoding="utf-8")


def combine(out_dir=REPORT_DIR) -> None:
    """arm 비교표. 게이트 판단 컬럼은 corpus/*(실음성) 기준 — 스펙 §1·§4."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for p in sorted(out_dir.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            reports.append(json.load(f))
    if not reports:
        print(f"[eval] 비교할 리포트 없음: {out_dir}/*.json — 먼저 evaluate 를 실행")
        return
    lines = ["# baseline arm 비교", "",
             "게이트 판단은 corpus(실음성) 컬럼 기준. 키워드 히트율은 TTS 긴급문구.", "",
             "| arm | CER(전체) | corpus clean | corpus noise@5 | corpus noise@0 "
             "| 키워드 히트율 | loudspeaker CER 평균 | n |",
             "|---|---|---|---|---|---|---|---|"]
    for r in reports:
        cs = r.get("by_condition_source", {})
        ls = [m["cer"] for k, m in r["by_condition"].items()
              if k.startswith("loudspeaker") and m["cer"] is not None]
        ls_avg = sum(ls) / len(ls) if ls else None
        lines.append(
            f"| {r['arm']} | {_fmt(r['overall']['cer'])} "
            f"| {_fmt(cs.get('corpus/clean', {}).get('cer'))} "
            f"| {_fmt(cs.get('corpus/noise@5', {}).get('cer'))} "
            f"| {_fmt(cs.get('corpus/noise@0', {}).get('cer'))} "
            f"| {_fmt(r['keyword_hit_rate'])} | {_fmt(ls_avg)} "
            f"| {r['overall']['n']} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((out_dir / "summary.md").read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="평가셋 v0 baseline 측정")
    ap.add_argument("--model", default="small",
                    help="small | medium | large-v3-turbo | 로컬 CT2 경로 "
                         "(WhisperModel 은 경로도 받는다 — Jetson 용)")
    ap.add_argument("--bias", action="store_true", help="initial_prompt+hotwords arm")
    ap.add_argument("--limit", type=int, default=None, help="앞 N개만(스모크)")
    ap.add_argument("--combine", action="store_true", help="리포트 비교표만 생성")
    args = ap.parse_args()

    if args.combine:
        combine()
        return

    # 경로 입력(예: ./ct2-ko)이어도 리포트 파일명이 유효하도록 마지막 요소만 사용
    arm = Path(args.model).name + ("-bias" if args.bias else "")
    rep = run_eval(MANIFEST, REPORT_DIR, arm,
                   _make_whisper_fn(args.model, args.bias), limit=args.limit)
    print(f"[eval] {arm}: CER {_fmt(rep['overall']['cer'])}, "
          f"키워드 히트율 {_fmt(rep['keyword_hit_rate'])}")


if __name__ == "__main__":
    main()
