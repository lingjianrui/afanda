# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Shared engine builders used by local CLI entry points.

Returns `(engine_config, engine_run)` for each supported engine kind, where
`engine_run` is the worklet-shaped callable `(EventBus, StreamClocks) -> Coroutine`.

Credentials are supplied inline via `EngineOptions` so the local UI can pass
them per-session instead of relying on process env vars.
"""

import base64
import os
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, Discriminator

from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.conversation_engines.cartesia_client import CartesiaApiClient
from avaturn_live_streamer.conversation_engines.configs import (
    CartesiaConversationEngineConfig,
    ConversationEngineConfig,
    OpenAIRealtimeAPIConversationEngineConfig,
    OpenaiRealtimeApiVoice,
    QwenOmniRealtimeConversationEngineConfig,
)
from avaturn_live_streamer.conversation_engines.qwen_omni_realtime_client import (
    QwenOmniRealtimeClient,
)
from avaturn_live_streamer.conversation_engines.voice_enrollment import (
    QwenVoiceMode,
    create_cloned_voice,
    enrollment_url_for_realtime_endpoint,
    mime_type_for_filename,
)
from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer.conversation_engines.realtime_api_client import RealtimeApiClient
from avaturn_live_streamer.event_bus import EventBus

EngineKind = Literal["openai", "cartesia", "qwen"]
ENGINE_KINDS: tuple[EngineKind, ...] = ("openai", "cartesia", "qwen")

type EngineRun = Callable[[EventBus, StreamClocks], Coroutine[None, None, None]]
type BuiltEngine = tuple[ConversationEngineConfig, EngineRun]

_CARTESIA_TOKEN_URL = "https://api.cartesia.ai/access-token"
_CARTESIA_VERSION = "2025-04-16"

DEFAULT_OPENAI_PROMPT = (
    "You are a friendly, concise voice assistant. Speak naturally and keep "
    "answers under 50 words. Avoid emojis or unreadable symbols."
)
DEFAULT_OPENAI_VOICE: OpenaiRealtimeApiVoice = "shimmer"
DEFAULT_OPENAI_MODEL = "gpt-realtime"
_OPENAI_MODEL_ALIASES = {
    "gpt-realtime-2": DEFAULT_OPENAI_MODEL,
}
DEFAULT_QWEN_ENDPOINT = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
DEFAULT_QWEN_MODEL = "qwen3-omni-flash-realtime"
DEFAULT_QWEN_CLONE_MODEL = "qwen3.5-omni-plus-realtime"
DEFAULT_QWEN_VOICE = "Cherry"

_LOGGER = get_logger()
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _normalize_openai_model(model: str) -> str:
    model = model.strip() or DEFAULT_OPENAI_MODEL
    return _OPENAI_MODEL_ALIASES.get(model, model)


class OpenAIEngineOptions(BaseModel):
    type: Literal["openai"] = "openai"
    api_key: str
    base_url: str | None = None
    model: str = DEFAULT_OPENAI_MODEL
    prompt: str = DEFAULT_OPENAI_PROMPT
    voice: OpenaiRealtimeApiVoice = DEFAULT_OPENAI_VOICE


class CartesiaEngineOptions(BaseModel):
    type: Literal["cartesia"] = "cartesia"
    api_key: str
    agent_id: str


class QwenEngineOptions(BaseModel):
    type: Literal["qwen"] = "qwen"
    api_key: str
    endpoint: str = DEFAULT_QWEN_ENDPOINT
    model: str = DEFAULT_QWEN_MODEL
    voice_mode: QwenVoiceMode = "preset"
    voice: str = DEFAULT_QWEN_VOICE
    voice_clone_name: str | None = None
    voice_clone_audio_b64: str | None = None
    voice_clone_mime_type: str | None = None
    voice_clone_file_path: str | None = None
    prompt: str = DEFAULT_OPENAI_PROMPT
    enable_vision: bool = False


EngineOptions = Annotated[
    OpenAIEngineOptions | CartesiaEngineOptions | QwenEngineOptions,
    Discriminator("type"),
]


async def _mint_cartesia_token(api_key: str) -> str:
    async with httpx.AsyncClient(timeout=10.0) as http:
        r = await http.post(
            _CARTESIA_TOKEN_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Cartesia-Version": _CARTESIA_VERSION,
            },
            json={"grants": {"agent": True}, "expires_in": 300},
        )
        r.raise_for_status()
        token = r.json().get("token")
        if not token:
            raise RuntimeError("Cartesia access-token response missing 'token' field")
        return token


async def mint_openai_realtime_secret(
    *,
    api_key: str,
    base_url: str | None = None,
    model: str = DEFAULT_OPENAI_MODEL,
    prompt: str = DEFAULT_OPENAI_PROMPT,
    voice: OpenaiRealtimeApiVoice = DEFAULT_OPENAI_VOICE,
    tracing: dict[str, object] | str | None = "auto",
) -> str:
    from openai import AsyncClient

    client_kwargs: dict[str, str] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    oai = AsyncClient(**client_kwargs)
    session: dict[str, object] = {
        "type": "realtime",
        "model": model,
        "audio": {
            "input": {
                "turn_detection": {"type": "semantic_vad", "eagerness": "high"},
            },
            "output": {"voice": voice},
        },
    }
    if prompt.strip():
        session["instructions"] = prompt
    if tracing is not None:
        session["tracing"] = tracing
    secret = await oai.realtime.client_secrets.create(
        expires_after={"seconds": 7200, "anchor": "created_at"},
        session=session,  # pyright: ignore [reportArgumentType]
    )
    return secret.value


async def build_cartesia(
    *,
    stream_id: str,
    options: CartesiaEngineOptions,
) -> BuiltEngine:
    token = await _mint_cartesia_token(options.api_key)
    cfg = CartesiaConversationEngineConfig(access_token=token, agent_id=options.agent_id)
    return cfg, CartesiaApiClient(cfg, stream_id=stream_id).run


async def build_openai(
    *,
    stream_id: str,
    options: OpenAIEngineOptions,
) -> BuiltEngine:
    tracing: dict[str, object] = {
        "workflow_name": "avaturn-live-local",
        "group_id": stream_id,
        "metadata": {"engine": "openai-realtime", "stream_id": stream_id},
    }
    secret = await mint_openai_realtime_secret(
        api_key=options.api_key,
        base_url=options.base_url,
        model=_normalize_openai_model(options.model),
        prompt=options.prompt,
        voice=options.voice,
        tracing=tracing,
    )
    cfg = OpenAIRealtimeAPIConversationEngineConfig(
        client_secret=secret,
        base_url=options.base_url,
    )
    return cfg, RealtimeApiClient(cfg).run


def _resolve_voice_clone_file(path_str: str) -> tuple[bytes, str]:
    path = Path(path_str)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"voice clone audio not found: {path}")
    return path.read_bytes(), mime_type_for_filename(path.name)


def qwen_env_defaults() -> dict[str, str | bool]:
    """Optional Qwen defaults from env for ``pixi run interactive-demo`` + ``.env``."""
    keys = {
        "voice_mode": "QWEN__VOICE_MODE",
        "voice": "QWEN__VOICE",
        "model": "QWEN__MODEL",
        "endpoint": "QWEN__ENDPOINT",
        "voice_clone_name": "QWEN__VOICE_CLONE_NAME",
        "voice_clone_file_path": "QWEN__VOICE_CLONE_FILE",
        "enable_vision": "QWEN__ENABLE_VISION",
    }
    out: dict[str, str | bool] = {}
    for field, env_name in keys.items():
        if val := os.environ.get(env_name, "").strip():
            if field == "enable_vision":
                out[field] = val.lower() in ("1", "true", "yes", "on")
            else:
                out[field] = val
    return out


def apply_qwen_env_defaults(options: QwenEngineOptions) -> QwenEngineOptions:
    updates = qwen_env_defaults()
    if not updates:
        return options
    return options.model_copy(update=updates)


async def _resolve_qwen_voice(options: QwenEngineOptions) -> str:
    endpoint = options.endpoint.rstrip("/") or DEFAULT_QWEN_ENDPOINT
    model = options.model.strip() or DEFAULT_QWEN_MODEL

    match options.voice_mode:
        case "preset":
            return options.voice.strip() or DEFAULT_QWEN_VOICE
        case "cloned":
            voice_id = options.voice.strip()
            if not voice_id:
                raise ValueError("cloned voice id is required when voice_mode is 'cloned'")
            return voice_id
        case "enroll":
            clone_name = (options.voice_clone_name or "").strip()
            audio_b64 = (options.voice_clone_audio_b64 or "").strip()
            mime_type = (options.voice_clone_mime_type or "").strip()
            file_path = (options.voice_clone_file_path or "").strip()
            if not clone_name:
                raise ValueError("voice_clone_name is required when voice_mode is 'enroll'")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                if not mime_type:
                    raise ValueError(
                        "voice_clone_mime_type is required when voice_clone_audio_b64 is set"
                    )
            elif file_path:
                audio_bytes, mime_type = _resolve_voice_clone_file(file_path)
            else:
                raise ValueError(
                    "voice_clone_audio_b64 or voice_clone_file_path is required "
                    "when voice_mode is 'enroll'"
                )
            voice_id = await create_cloned_voice(
                api_key=options.api_key,
                target_model=model,
                preferred_name=clone_name,
                audio_bytes=audio_bytes,
                audio_mime_type=mime_type,
                enrollment_url=enrollment_url_for_realtime_endpoint(endpoint),
            )
            _LOGGER.info("qwen voice enrolled", voice=voice_id, preferred_name=clone_name)
            return voice_id
        case _:
            raise ValueError(f"unsupported voice_mode: {options.voice_mode}")


async def build_qwen(
    *,
    stream_id: str,
    options: QwenEngineOptions,
) -> BuiltEngine:
    _ = stream_id
    voice = await _resolve_qwen_voice(options)
    cfg = QwenOmniRealtimeConversationEngineConfig(
        api_key=options.api_key,
        endpoint=options.endpoint.rstrip("/") or DEFAULT_QWEN_ENDPOINT,
        model=options.model.strip() or DEFAULT_QWEN_MODEL,
        voice=voice,
        instructions=options.prompt,
        enable_vision=options.enable_vision,
    )
    return cfg, QwenOmniRealtimeClient(cfg).run


async def build_engine(options: EngineOptions, *, stream_id: str) -> BuiltEngine:
    match options:
        case OpenAIEngineOptions():
            return await build_openai(stream_id=stream_id, options=options)
        case CartesiaEngineOptions():
            return await build_cartesia(stream_id=stream_id, options=options)
        case QwenEngineOptions():
            return await build_qwen(stream_id=stream_id, options=options)
