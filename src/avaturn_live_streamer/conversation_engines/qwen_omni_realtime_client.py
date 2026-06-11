# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Qwen-Omni-Realtime WebSocket client with event bus integration."""

import json
from asyncio import TaskGroup
from base64 import b64decode, b64encode
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import attrs
import websockets
from websockets.asyncio.client import ClientConnection

from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.conversation_engines.configs import (
    QwenOmniRealtimeConversationEngineConfig,
)
from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import (
    DiscardAvatarSpeechBuffer,
    InputTranscript,
    ResponseTranscript,
    SegmentChunkGenerated,
    SegmentGenerationCompleted,
    SegmentGenerationStarted,
    Shutdown,
    TextEchoEnqueueText,
    UserSpeechReceived,
    UserVisionFrameReceived,
)
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer
from avaturn_live_streamer.utils.async_utils import run_in_thread
from avaturn_live_streamer.utils.datetime import tzutcnow
from avaturn_live_streamer.utils.exceptions import async_log_entry_exit

_LOGGER = get_logger()

_QWEN_INPUT_SAMPLE_RATE = 16_000
_QWEN_OUTPUT_SAMPLE_RATE = 24_000


def _event_id() -> str:
    return f"event_{uuid4().hex}"


def _qwen_ws_url(endpoint: str, model: str) -> str:
    parsed = urlsplit(endpoint.rstrip("/"))
    query = dict(parse_qsl(parsed.query))
    query["model"] = model
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query),
            "",
        )
    )


@attrs.define
class QwenOmniRealtimeClient:
    _config: QwenOmniRealtimeConversationEngineConfig
    _current_response_id: str | None = None
    _has_sent_audio: bool = False

    @asynccontextmanager
    async def _connect(self) -> AsyncGenerator[ClientConnection, None]:
        async with websockets.connect(
            _qwen_ws_url(self._config.endpoint, self._config.model),
            additional_headers={
                "Authorization": f"Bearer {self._config.api_key}",
            },
        ) as ws:
            await ws.send(
                json.dumps(
                    {
                        "event_id": _event_id(),
                        "type": "session.update",
                        "session": {
                            "modalities": ["text", "audio"],
                            "voice": self._config.voice,
                            "input_audio_format": "pcm",
                            "output_audio_format": "pcm",
                            "instructions": self._config.instructions,
                            "input_audio_transcription": {
                                "model": "gummy-realtime-v1",
                            },
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": 0.5,
                                "silence_duration_ms": 800,
                                "create_response": True,
                                "interrupt_response": True,
                            },
                        },
                    }
                )
            )
            yield ws

    async def _cancel_current_response(self, ws: ClientConnection) -> None:
        if self._current_response_id is None:
            return
        await ws.send(
            json.dumps(
                {
                    "event_id": _event_id(),
                    "type": "response.cancel",
                }
            )
        )
        self._current_response_id = None

    async def _send_speech(self, ws: ClientConnection, buffer: SpeechBuffer) -> None:
        audio = buffer.resample(_QWEN_INPUT_SAMPLE_RATE).to_bytes()
        await ws.send(
            json.dumps(
                {
                    "event_id": _event_id(),
                    "type": "input_audio_buffer.append",
                    "audio": b64encode(audio).decode(),
                }
            )
        )
        self._has_sent_audio = True

    async def _send_vision_frame(self, ws: ClientConnection, jpeg: bytes) -> None:
        if not self._config.enable_vision:
            return
        if not self._has_sent_audio:
            _LOGGER.debug("Skipping vision frame until audio has been sent")
            return
        await ws.send(
            json.dumps(
                {
                    "event_id": _event_id(),
                    "type": "input_image_buffer.append",
                    "image": b64encode(jpeg).decode(),
                }
            )
        )

    def _decode_chunk(self, delta_base64: str) -> SpeechBuffer:
        chunk = SpeechBuffer.from_bytes(
            b64decode(delta_base64),
            _QWEN_OUTPUT_SAMPLE_RATE,
        )
        _LOGGER.debug("Received Qwen chunk with duration=%.3f", float(chunk.duration))
        return chunk

    async def _listener(self, bus: EventBus, ws: ClientConnection) -> None:
        conv_items_started_by_id: dict[str, bool] = {}
        bus.ready()
        async for data in ws:
            msg = await run_in_thread(json.loads, data)
            msg_type = msg.get("type")

            if msg_type == "error":
                _LOGGER.error("Received Qwen error", error=msg.get("error"))
                raise RuntimeError(msg.get("error"))

            msg_to_log = dict(msg)
            msg_to_log.pop("delta", None)
            _LOGGER.debug("Received Qwen message", message=msg_to_log)

            match msg_type:
                case "response.created":
                    self._current_response_id = msg["response"]["id"]
                case "response.done":
                    response = msg.get("response", {})
                    if self._current_response_id == response.get("id"):
                        self._current_response_id = None
                case "response.audio.delta":
                    item_id = (
                        msg.get("item_id") or self._current_response_id or "qwen-response"
                    )
                    chunk = await run_in_thread(self._decode_chunk, msg["delta"])
                    if not conv_items_started_by_id.get(item_id):
                        conv_items_started_by_id[item_id] = True
                        await bus.publish(SegmentGenerationStarted(segment_id=item_id))
                    await bus.publish(SegmentChunkGenerated(segment_id=item_id, buffer=chunk))
                case "response.audio.done":
                    item_id = (
                        msg.get("item_id") or self._current_response_id or "qwen-response"
                    )
                    if conv_items_started_by_id.get(item_id):
                        await bus.publish(SegmentGenerationCompleted(segment_id=item_id))
                        del conv_items_started_by_id[item_id]
                case "input_audio_buffer.speech_started":
                    _LOGGER.info("Qwen speech interrupted")
                    await self._cancel_current_response(ws)
                    await bus.publish(DiscardAvatarSpeechBuffer())
                case "conversation.item.input_audio_transcription.completed":
                    await bus.publish(
                        InputTranscript(
                            transcript=msg.get("transcript", ""),
                            timestamp=tzutcnow().timestamp(),
                        )
                    )
                case "conversation.item.input_audio_transcription.failed":
                    await bus.publish(
                        InputTranscript(
                            transcript="transcription-failed",
                            timestamp=tzutcnow().timestamp(),
                        )
                    )
                case "response.audio_transcript.done":
                    await bus.publish(
                        ResponseTranscript(
                            transcript=msg.get("transcript", ""),
                            timestamp=tzutcnow().timestamp(),
                        )
                    )
                case _:
                    pass

    async def _handle_bus_events(self, bus: EventBus, ws: ClientConnection) -> None:
        event_types = [TextEchoEnqueueText, UserSpeechReceived, Shutdown]
        if self._config.enable_vision:
            event_types.append(UserVisionFrameReceived)
        async with bus.subscribe(*event_types) as sub:
            bus.ready()
            async for event in sub:
                match event:
                    case TextEchoEnqueueText(text=txt):
                        _LOGGER.warning(
                            "TextEchoEnqueueText is not supported by Qwen Omni Realtime",
                            text=txt,
                        )
                    case UserSpeechReceived(buffer=buf):
                        await self._send_speech(ws, buf)
                    case UserVisionFrameReceived(jpeg=jpeg):
                        await self._send_vision_frame(ws, jpeg)
                    case Shutdown():
                        await ws.close()
                        return

    @async_log_entry_exit
    async def run(self, bus: EventBus, clocks: StreamClocks) -> None:
        """Connect to Qwen-Omni-Realtime WebSocket and process messages."""
        _ = clocks
        async with self._connect() as ws:
            async with TaskGroup() as tg:
                tg.create_task(
                    self._listener(bus.clone(), ws),
                    name="QwenOmniRealtimeClient._listener",
                )
                tg.create_task(
                    self._handle_bus_events(bus.clone(), ws),
                    name="QwenOmniRealtimeClient._handle_bus_events",
                )
                bus.ready()
