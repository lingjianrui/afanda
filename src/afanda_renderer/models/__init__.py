# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Per-model I/O dataclasses and engine type aliases.

One file per model. Each module defines a ``${Name}Input`` and
``${Name}Output`` dataclass whose field names match the underlying engine's
tensor names exactly, plus a ``${Name}Engine`` type alias for
``InferenceEngine[Input, Output]``. Pre/post-processing helpers live with
the call sites in ``afanda_renderer.components`` -- this package is the
contract layer only.
"""

from afanda_renderer.models.appearance_extractor import (
    AEInput,
    AEOutput,
    AppearanceExtractorEngine,
)
from afanda_renderer.models.decoder import DecoderEngine, DecoderInput, DecoderOutput
from afanda_renderer.models.face_detection import (
    FaceDetEngine,
    FaceDetInput,
    FaceDetOutput,
)
from afanda_renderer.models.hubert import HubertEngine, HubertInput, HubertOutput
from afanda_renderer.models.landmark106 import Lm106Engine, Lm106Input, Lm106Output
from afanda_renderer.models.landmark203 import Lm203Engine, Lm203Input, Lm203Output
from afanda_renderer.models.afanda import (
    AfandaDecodeEngine,
    AfandaDecodeInput,
    AfandaDecodeOutput,
    AfandaEncodeEngine,
    AfandaEncodeInput,
    AfandaEncodeOutput,
)
from afanda_renderer.models.matting import MODNetEngine, MODNetInput, MODNetOutput
from afanda_renderer.models.motion_extractor import (
    MotionExtractorEngine,
    MotionInput,
    MotionOutput,
)
from afanda_renderer.models.stitch import StitchEngine, StitchInput, StitchOutput
from afanda_renderer.models.warp import WarpEngine, WarpInput, WarpOutput

__all__ = [
    "AEInput",
    "AEOutput",
    "AppearanceExtractorEngine",
    "DecoderEngine",
    "DecoderInput",
    "DecoderOutput",
    "FaceDetEngine",
    "FaceDetInput",
    "FaceDetOutput",
    "HubertEngine",
    "HubertInput",
    "HubertOutput",
    "Lm106Engine",
    "Lm106Input",
    "Lm106Output",
    "Lm203Engine",
    "Lm203Input",
    "Lm203Output",
    "AfandaDecodeEngine",
    "AfandaDecodeInput",
    "AfandaDecodeOutput",
    "AfandaEncodeEngine",
    "AfandaEncodeInput",
    "AfandaEncodeOutput",
    "MODNetEngine",
    "MODNetInput",
    "MODNetOutput",
    "MotionExtractorEngine",
    "MotionInput",
    "MotionOutput",
    "StitchEngine",
    "StitchInput",
    "StitchOutput",
    "WarpEngine",
    "WarpInput",
    "WarpOutput",
]
