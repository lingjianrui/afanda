# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from typing import Literal

from pydantic import BaseModel, ConfigDict

OpenaiRealtimeApiVoice = Literal[
    "alloy",
    "ash",
    "ballad",
    "cedar",
    "coral",
    "echo",
    "marin",
    "sage",
    "shimmer",
    "verse",
]


class OpenAIRealtimeAPIConversationEngineConfig(BaseModel):
    type: Literal["openai-realtime"] = "openai-realtime"
    client_secret: str | None = None
    base_url: str | None = None

    model_config = ConfigDict(title="OpenAIRealtimeAPICEConfig")


class CartesiaConversationEngineConfig(BaseModel):
    type: Literal["cartesia"] = "cartesia"
    access_token: str
    agent_id: str

    model_config = ConfigDict(title="CartesiaCEConfig")


class QwenOmniRealtimeConversationEngineConfig(BaseModel):
    type: Literal["qwen-omni-realtime"] = "qwen-omni-realtime"
    api_key: str
    endpoint: str
    model: str
    voice: str
    instructions: str
    enable_vision: bool = False

    model_config = ConfigDict(title="QwenOmniRealtimeCEConfig")


ConversationEngineConfig = (
    OpenAIRealtimeAPIConversationEngineConfig
    | CartesiaConversationEngineConfig
    | QwenOmniRealtimeConversationEngineConfig
)
