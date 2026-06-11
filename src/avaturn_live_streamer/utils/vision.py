# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Helpers for Qwen-Omni vision input (JPEG frames from WebRTC camera)."""

import av

# Qwen recommends <=190 KB raw before Base64; keep a small margin.
_MAX_JPEG_BYTES = 190_000


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
        return None

    codec.width = frame.width
    codec.height = frame.height
    codec.pix_fmt = "yuvj420p"
    codec.flags |= av.codec.context.Flags.QSCALE
    qscale = max(1, min(31, int((100 - quality) * 31 / 100)))
    codec.qmin = codec.qmax = qscale

    yuvj = frame.reformat(format="yuvj420p")
    chunks: list[bytes] = []
    for packet in codec.encode(yuvj):
        if packet:
            chunks.append(bytes(packet))
    for packet in codec.encode(None):
        if packet:
            chunks.append(bytes(packet))
    if not chunks:
        return None

    data = b"".join(chunks)
    if len(data) > _MAX_JPEG_BYTES:
        if max_dim > 480:
            return video_frame_to_jpeg(frame, max_dim=480, quality=quality)
        if quality > 40:
            return video_frame_to_jpeg(frame, max_dim=max_dim, quality=quality - 15)
        return None
    return data
