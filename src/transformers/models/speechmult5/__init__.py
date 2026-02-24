# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_torch_available,
)


_import_structure = {
    "configuration_speechmult5": [
        "SpeechMult5Config",
        "SpeechMult5HifiGanConfig",
    ],
    "feature_extraction_speechmult5": ["SpeechMult5FeatureExtractor"],
    "processing_speechmult5": ["SpeechMult5Processor"],
}

try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_speechmult5"] = ["SpeechMult5Tokenizer"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_speechmult5"] = [
        "SpeechMult5ForSpeechToText",
        "SpeechMult5ForSpeechToSpeech",
        "SpeechMult5ForTextToSpeech",
        "SpeechMult5Model",
        "SpeechMult5PreTrainedModel",
        "SpeechMult5HifiGan",
    ]

if TYPE_CHECKING:
    from .configuration_speechmult5 import (
        SpeechMult5Config,
        SpeechMult5HifiGanConfig,
    )
    from .feature_extraction_speechmult5 import SpeechMult5FeatureExtractor
    from .processing_speechmult5 import SpeechMult5Processor

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_speechmult5 import SpeechMult5Tokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_speechmult5 import (
            SpeechMult5ForSpeechToSpeech,
            SpeechMult5ForSpeechToText,
            SpeechMult5ForTextToSpeech,
            SpeechMult5HifiGan,
            SpeechMult5Model,
            SpeechMult5PreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
