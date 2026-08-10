#!/usr/bin/env bash
# 젯슨 원커맨드 사이렌 수집: ./collect.sh [장소]
#
# emergency-hud 상주 서비스가 ReSpeaker 를 잡고 있으면 잠시 내리고(마이크 해제),
# 수집 모드 HUD(감지 자동 녹음 + 차종 버튼 라벨 + 미검출 수동 녹음)를 띄운 뒤,
# 종료(ESC/Q)하면 서비스를 원래대로 되살린다.
#
# 채널·장치는 main.py 가 자동(ReSpeaker → 6ch), 세션은 data/collect_sessions/ 에 쌓인다.
# 수집 뒤 노트북에서: 세션 폴더 복사 → python tools/cut_collect.py
set -u
cd "$(dirname "$0")"
PLACE="${1:-미지정}"

# ssh 로 붙어 실행해도 젯슨에 물린 화면으로 HUD 가 뜨게
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

RESTORE=0
if systemctl is-active --quiet emergency-hud 2>/dev/null; then
    echo "[collect] emergency-hud 서비스 정지 (ReSpeaker 해제)"
    if ! sudo systemctl stop emergency-hud; then
        echo "[collect] 서비스 정지 실패 — sudo 권한 필요" >&2
        exit 1
    fi
    RESTORE=1
fi
restore() {
    if [ "$RESTORE" = 1 ]; then
        echo "[collect] emergency-hud 서비스 재시작"
        sudo systemctl start emergency-hud || true
    fi
}
trap restore EXIT

echo "[collect] 장소: $PLACE — 종료는 HUD 에서 ESC 또는 Q"
"$PY" main.py --mic --hud --collect --place "$PLACE"
