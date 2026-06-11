# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Helpers for Qwen-Omni vision input (JPEG frames from WebRTC camera)."""

import av

from avaturn_live_streamer.core.logs import get_logger

_LOGGER = get_logger()

# Qwen recommends <=190 KB raw before Base64; keep a small margin.
_MAX_JPEG_BYTES = 190_000


def _quality_to_qscale(quality: int) -> int:
    return max(1, min(31, int((100 - quality) * 31 / 100)))


def _configure_mjpeg_encoder(codec: av.codec.context.CodecContext, quality: int) -> None:
    """Apply JPEG quality settings across PyAV / FFmpeg versions."""
    qscale = _quality_to_qscale(quality)

    if hasattr(codec, "global_quality"):
        codec.global_quality = qscale
        return

    try:
        flags = av.codec.context.Flags
        if hasattr(flags, "QSCALE"):
            codec.flags |= flags.QSCALE
            codec.qmin = codec.qmax = qscale
            return
    except AttributeError:
        pass

    # PyAV 12+ on some builds exposes quality via codec options.
    for key in ("qscale", "q"):
        try:
            codec.options = {key: str(qscale)}
            return
        except Exception:
            continue


def _encode_mjpeg_frame(codec: av.codec.context.CodecContext, frame: av.VideoFrame) -> bytes | None:
    chunks: list[bytes] = []
    for packet in codec.encode(frame):
        if packet:
            chunks.append(bytes(packet))
    for packet in codec.encode(None):
        if packet:
            chunks.append(bytes(packet))
    if not chunks:
        return None
    return b"".join(chunks)


def video_frame_to_jpeg(
    frame: av.VideoFrame,
    *,
    max_dim: int = 720,
    quality: int = 80,
) -> bytes | None:
    """Encode an ``av.VideoFrame`` as JPEG within Qwen size limits."""
    w, h = frame.width, frame.height
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w = max(2, int(w * scale) & ~1)
        new_h = max(2, int(h * scale) & ~1)
        frame = frame.reformat(width=new_w, height=new_h)

    try:
        codec = av.codec.Codec("mjpeg", "w").create()
    except av.AVError:
        _LOGGER.warning("mjpeg encoder unavailable")
        return None

    codec.width = frame.width
    codec.height = frame.height
    codec.pix_fmt = "yuvj420p"
    _configure_mjpeg_encoder(codec, quality)

    try:
        if hasattr(codec, "is_open") and not codec.is_open:
            codec.open()
        yuvj = frame.reformat(format="yuvj420p")
        data = _encode_mjpeg_frame(codec, yuvj)
    except Exception:
        _LOGGER.exception("failed to encode vision frame as JPEG")
        return None

    if data is None:
        return None

    if len(data) > _MAX_JPEG_BYTES:
        if max_dim > 480:
            return video_frame_to_jpeg(frame, max_dim=480, quality=quality)
        if quality > 40:
            return video_frame_to_jpeg(frame, max_dim=max_dim, quality=quality - 15)
        _LOGGER.warning("vision JPEG still too large after downscale", size=len(data))
        return None
    return data
