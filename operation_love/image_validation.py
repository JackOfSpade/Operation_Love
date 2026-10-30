"""Bounded, complete PNG validation shared by device capture and the Training Hub.

Uses only the standard library so the ADB transport remains available in core-only installs.
"""

from __future__ import annotations

import struct
import zlib

_MAX_RASTER_DIMENSION = 10_000
_MAX_RASTER_PIXELS = 30_000_000
# The pixel-count ceiling still permits a 30 MP, 16-bit RGBA image whose inflated raster is
# 240 MB.  Device screenshots are 8-bit and far smaller; cap the actual zlib output too so a
# tiny, highly compressible frame cannot monopolize the Hub process while being validated.
_MAX_DECODED_PNG_BYTES = 64 * 1024 * 1024
_MAX_PNG_CHUNKS = 10_000


def _valid_png_chunk_type(kind: bytes) -> bool:
    """Whether a four-byte PNG chunk type follows PNG's letter/reserved-bit rules."""
    return (
        len(kind) == 4
        and all(65 <= byte <= 90 or 97 <= byte <= 122 for byte in kind)
        # PNG reserves the third chunk-type bit; compliant decoders require it uppercase.
        and 65 <= kind[2] <= 90
    )


def _valid_dimensions(width: int, height: int) -> bool:
    return (
        0 < width <= _MAX_RASTER_DIMENSION
        and 0 < height <= _MAX_RASTER_DIMENSION
        and width * height <= _MAX_RASTER_PIXELS
    )


def _valid_png_idat(
    *,
    width: int,
    height: int,
    bit_depth: int,
    color_type: int,
    compression: int,
    filter_method: int,
    interlace: int,
    data: bytes,
) -> bool:
    """Prove that a bounded, non-interlaced PNG has decodable pixel rows."""
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    allowed_depths = {
        0: {1, 2, 4, 8, 16},
        2: {8, 16},
        3: {1, 2, 4, 8},
        4: {8, 16},
        6: {8, 16},
    }
    if (
        channels is None
        or bit_depth not in allowed_depths[color_type]
        or compression != 0
        or filter_method != 0
        or interlace != 0
    ):
        return False
    row_bytes = (width * channels * bit_depth + 7) // 8
    expected = height * (row_bytes + 1)
    if expected > _MAX_DECODED_PNG_BYTES:
        return False
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(data, expected + 1)
        if (
            len(raw) != expected
            or not decoder.eof
            or decoder.unconsumed_tail
            or decoder.unused_data
        ):
            return False
        # A zlib stream with the expected number of bytes is not necessarily a PNG raster: every
        # scanline starts with one of PNG's five reconstruction filter bytes.  The old size-only
        # check accepted (for example) filter byte 5, which is CRC-valid and decompresses cleanly
        # but browsers cannot render.  A Hub checkpoint with that frame looked reviewable while
        # its image endpoint was broken, so Like/Dislike must stay disabled at publication time.
        return all(raw[row * (row_bytes + 1)] <= 4 for row in range(height))
    except zlib.error:
        return False


def _raster_mime(image: object) -> str | None:
    """Recognize only a complete, bounded, browser-decodable PNG review frame.

    Android's production ``screencap -p`` output is PNG.  Reject JPEG/WebP rather than accept
    a merely structural envelope: validating their compressed payloads safely would require a
    real bounded decoder, and a broken Hub image must never leave Like/Dislike enabled.
    """
    if not isinstance(image, bytes) or not image:
        return None
    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        offset = 8
        chunks = 0
        saw_idat = False
        finished_idat = False
        saw_palette = False
        idat = []
        png_header = None
        while offset + 12 <= len(image) and chunks < _MAX_PNG_CHUNKS:
            size = struct.unpack(">I", image[offset : offset + 4])[0]
            kind = image[offset + 4 : offset + 8]
            data_start = offset + 8
            end = data_start + size + 4
            if end > len(image):
                return None
            data = image[data_start : data_start + size]
            expected_crc = struct.unpack(">I", image[data_start + size : end])[0]
            if zlib.crc32(kind + data) & 0xFFFFFFFF != expected_crc:
                return None
            if not _valid_png_chunk_type(kind):
                return None
            if chunks == 0:
                if kind != b"IHDR" or size != 13:
                    return None
                (
                    width,
                    height,
                    bit_depth,
                    color_type,
                    compression,
                    filter_method,
                    interlace,
                ) = struct.unpack(">IIBBBBB", data)
                if not _valid_dimensions(width, height):
                    return None
                png_header = (
                    width,
                    height,
                    bit_depth,
                    color_type,
                    compression,
                    filter_method,
                    interlace,
                )
            elif kind == b"IHDR":
                return None
            elif kind == b"PLTE":
                # Indexed-colour PNGs are meaningless without their palette.  Validate the
                # palette here, while chunk order is available, rather than accepting a
                # structurally complete-but-undecodable frame in `_valid_png_idat` below.
                assert png_header is not None
                if (
                    png_header[3] in {0, 4}
                    or saw_idat
                    or saw_palette
                    or size == 0
                    or size % 3
                ):
                    return None
                entries = size // 3
                # PLTE is capped at 256 entries for every colour type. Indexed PNGs have the
                # additional bit-depth cap (a 1-bit image has no use for 256 palette entries).
                if entries > 256 or (
                    png_header[3] == 3 and entries > 2 ** png_header[2]
                ):
                    return None
                saw_palette = True
            elif kind not in {b"IDAT", b"IEND"} and not (kind[0] & 0x20):
                # Unknown critical chunks make the image undecodable by definition. Ancillary
                # chunks (lowercase first byte) may safely be ignored by a browser, but treating
                # an unrecognised required chunk as metadata would publish a broken checkpoint.
                return None
            if kind == b"IDAT":
                if finished_idat or (
                    png_header is not None and png_header[3] == 3 and not saw_palette
                ):
                    return None
                saw_idat = True
                idat.append(data)
            elif saw_idat:
                # IDAT chunks form one contiguous compressed stream.  Joining chunks separated
                # by another chunk can decompress even though a browser must reject the PNG.
                finished_idat = True
            if kind == b"IEND":
                if size != 0 or not saw_idat or end != len(image) or png_header is None:
                    return None
                return (
                    "image/png"
                    if _valid_png_idat(
                        width=png_header[0],
                        height=png_header[1],
                        bit_depth=png_header[2],
                        color_type=png_header[3],
                        compression=png_header[4],
                        filter_method=png_header[5],
                        interlace=png_header[6],
                        data=b"".join(idat),
                    )
                    else None
                )
            offset, chunks = end, chunks + 1
        return None
    return None
