# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Qwen voice cloning (voice enrollment) for realtime Omni models."""

import base64
from typing import Literal

import httpx

_MIME_BY_SUFFIX = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
}

DEFAULT_ENROLLMENT_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"
)
INTL_ENROLLMENT_URL = (
    "https://dashscope-intl.aliyuncs.com/api/v1/services/audio/tts/customization"
)
VOICE_ENROLLMENT_MODEL = "qwen-voice-enrollment"


def enrollment_url_for_realtime_endpoint(endpoint: str) -> str:
    if "dashscope-intl" in endpoint:
        return INTL_ENROLLMENT_URL
    return DEFAULT_ENROLLMENT_URL


def mime_type_for_filename(filename: str) -> str:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mime = _MIME_BY_SUFFIX.get(f".{suffix}")
    if mime is None:
        raise ValueError(f"不支持的音频格式: .{suffix}，请使用 wav/mp3/m4a")
    return mime


async def create_cloned_voice(
    *,
    api_key: str,
    target_model: str,
    preferred_name: str,
    audio_bytes: bytes,
    audio_mime_type: str,
    enrollment_url: str | None = None,
    timeout: float = 60.0,
) -> str:
    """Upload reference audio and return the cloned ``voice`` id for realtime use."""
    if not preferred_name.strip():
        raise ValueError("preferred_name is required for voice enrollment")
    if not audio_bytes:
        raise ValueError("reference audio is empty")

    url = enrollment_url or DEFAULT_ENROLLMENT_URL
    data_uri = (
        f"data:{audio_mime_type};base64,"
        f"{base64.b64encode(audio_bytes).decode()}"
    )
    payload = {
        "model": VOICE_ENROLLMENT_MODEL,
        "input": {
            "action": "create",
            "target_model": target_model,
            "preferred_name": preferred_name.strip(),
            "audio": {"data": data_uri},
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=timeout) as http:
        resp = await http.post(url, json=payload, headers=headers)

    if resp.status_code != 200:
        raise RuntimeError(
            f"voice enrollment failed: HTTP {resp.status_code}: {resp.text[:500]}"
        )

    try:
        return resp.json()["output"]["voice"]
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"failed to parse voice enrollment response: {exc}") from exc


QwenVoiceMode = Literal["preset", "cloned", "enroll"]
