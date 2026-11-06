"""Figshare 공개 사이렌/도로소음 ZIP에서 소수 WAV만 HTTP Range로 내려받는다.

전체 ZIP(약 1.09 GB)을 받지 않고 중앙 디렉터리와 선택한 멤버의 압축 구간만
요청한다. 내려받은 데이터는 ``data/`` 아래에 저장되어 git에 포함되지 않는다.

출처
----
M. Usaid et al., "Large-Scale Audio Dataset for Emergency Vehicle Sirens
and Road Noises", Figshare, CC0, doi:10.6084/m9.figshare.19291472.v2
"""

from __future__ import annotations

import argparse
import binascii
import json
import struct
import urllib.request
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ARTICLE_ID = 19_291_472
API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
ZIP_NAME = "Emergency Vehicle Sirens.zip"
ZIP_NAMES = {
    "siren": ZIP_NAME,
    "road": "Road Noises.zip",
}


@dataclass(frozen=True)
class ZipMember:
    name: str
    compressed_size: int
    uncompressed_size: int
    compression: int
    local_offset: int
    crc32: int


def _get(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    headers = {"User-Agent": "EmergencySoundAssist-evaluator/1.0"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=90) as response:
        data = response.read()
    if byte_range is not None:
        expected = byte_range[1] - byte_range[0] + 1
        if len(data) != expected:
            raise RuntimeError(f"Range 응답 길이 불일치: expected={expected}, got={len(data)}")
    return data


def _article_metadata() -> dict:
    return json.loads(_get(API_URL).decode("utf-8"))


def _eocd(zip_url: str, zip_size: int) -> tuple[int, int, int]:
    tail_size = min(zip_size, 65_536)
    tail = _get(zip_url, (zip_size - tail_size, zip_size - 1))
    pos = tail.rfind(b"PK\x05\x06")
    if pos < 0:
        raise RuntimeError("ZIP EOCD를 찾지 못했습니다")
    fields = struct.unpack_from("<4s4H2LH", tail, pos)
    entries = int(fields[4])
    cd_size = int(fields[5])
    cd_offset = int(fields[6])
    return entries, cd_size, cd_offset


def _members(zip_url: str, zip_size: int) -> list[ZipMember]:
    expected_entries, cd_size, cd_offset = _eocd(zip_url, zip_size)
    central = _get(zip_url, (cd_offset, cd_offset + cd_size - 1))
    out: list[ZipMember] = []
    pos = 0
    while pos + 46 <= len(central) and central[pos : pos + 4] == b"PK\x01\x02":
        h = struct.unpack_from("<4s6H3L5H2L", central, pos)
        compression = int(h[4])
        crc32 = int(h[7])
        compressed_size = int(h[8])
        uncompressed_size = int(h[9])
        name_len, extra_len, comment_len = map(int, h[10:13])
        local_offset = int(h[16])
        raw_name = central[pos + 46 : pos + 46 + name_len]
        name = raw_name.decode("utf-8", "replace")
        out.append(
            ZipMember(name, compressed_size, uncompressed_size, compression, local_offset, crc32)
        )
        pos += 46 + name_len + extra_len + comment_len
    if len(out) != expected_entries:
        raise RuntimeError(f"ZIP 멤버 수 불일치: expected={expected_entries}, parsed={len(out)}")
    return out


def _choose(members: Iterable[ZipMember], count: int) -> list[ZipMember]:
    candidates = [
        m
        for m in members
        if m.name.lower().endswith(".wav")
        and m.compression in (0, 8)
        and m.uncompressed_size >= 250_000
        and m.compressed_size <= 4_000_000
    ]
    if len(candidates) <= count:
        return candidates
    # 파일 번호 한 구간에 몰리지 않도록 전체 목록에서 균등 선택한다.
    idx = [round(i * (len(candidates) - 1) / (count - 1)) for i in range(count)]
    return [candidates[i] for i in idx]


def _download_member(zip_url: str, member: ZipMember) -> bytes:
    # 로컬 헤더는 파일명/extra 길이를 알아낼 만큼만 먼저 읽는다.
    header = _get(zip_url, (member.local_offset, member.local_offset + 511))
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"로컬 ZIP 헤더가 아닙니다: {member.name}")
    h = struct.unpack_from("<4s5H3L2H", header, 0)
    name_len, extra_len = int(h[-2]), int(h[-1])
    data_offset = member.local_offset + 30 + name_len + extra_len
    compressed = _get(
        zip_url,
        (data_offset, data_offset + member.compressed_size - 1),
    )
    if member.compression == 0:
        payload = compressed
    elif member.compression == 8:
        payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:  # pragma: no cover - _choose가 차단
        raise RuntimeError(f"지원하지 않는 ZIP 압축 방식 {member.compression}")
    if len(payload) != member.uncompressed_size:
        raise RuntimeError(f"압축 해제 크기 불일치: {member.name}")
    if (binascii.crc32(payload) & 0xFFFFFFFF) != member.crc32:
        raise RuntimeError(f"CRC 불일치: {member.name}")
    return payload


def download_samples(output_dir: Path, count: int = 12, zip_name: str = ZIP_NAME) -> Path:
    metadata = _article_metadata()
    record = next(f for f in metadata["files"] if f["name"] == zip_name)
    zip_url = record["download_url"]
    selected = _choose(_members(zip_url, int(record["size"])), count)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for i, member in enumerate(selected, 1):
        target = output_dir / Path(member.name).name
        print(f"[{i:02d}/{len(selected):02d}] {member.name} ({member.compressed_size / 1e6:.1f} MB)")
        if not target.exists() or target.stat().st_size != member.uncompressed_size:
            target.write_bytes(_download_member(zip_url, member))
        saved.append({**asdict(member), "local_path": str(target)})

    manifest = {
        "source": metadata["citation"],
        "doi": metadata["doi"],
        "license": metadata["license"],
        "article_url": metadata["url_public_html"],
        "zip_file": zip_name,
        "samples": saved,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=tuple(ZIP_NAMES), default="siren")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()
    if args.count < 2:
        parser.error("--count는 2 이상이어야 합니다")
    output = args.output or Path("data/public_sirens" if args.kind == "siren" else "data/public_road_noise")
    download_samples(output, args.count, ZIP_NAMES[args.kind])


if __name__ == "__main__":
    main()
