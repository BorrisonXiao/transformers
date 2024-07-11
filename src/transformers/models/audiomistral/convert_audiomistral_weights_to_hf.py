#!/usr/bin/env python3
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
import argparse
import json
from pathlib import Path

from transformers import (
    AudioMistralConfig,
    AudioMistralForCausalLM,
    MistralForCausalLM,
    WhisperForConditionalGeneration,
)


CONFIG = dict(
    architectures=["AudioMistralForCausalLM"],
    vocab_size=32768,
    num_mel_bins=128,
    encoder_layers=32,
    encoder_attention_heads=20,
    encoder_ffn_dim=5120,
    encoder_layerdrop=0.0,
    encoder_activation_function="gelu",
    is_encoder_decoder=False,
    encoder_d_model=1280,
    encoder_dropout=0.0,
    encoder_attention_dropout=0.0,
    encoder_activation_dropout=0.0,
    encoder_init_std=0.02,
    max_source_positions=1500,
    scale_embedding=False,
    apply_spec_augment=False,
    mask_time_prob=0.05,
    mask_time_length=10,
    mask_time_min_masks=2,
    mask_feature_prob=0.0,
    mask_feature_length=10,
    mask_feature_min_masks=0,
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    hidden_act="silu",
    max_position_embeddings=4096 * 32,
    initializer_range=0.02,
    rms_norm_eps=1e-5,
    use_cache=True,
    pad_token_id=None,
    bos_token_id=1,
    eos_token_id=2,
    tie_word_embeddings=False,
    rope_theta=1000000.0,
    sliding_window=4096,
    attention_dropout=0.0,
)


def convert(
    config: Path,
    output_dir: Path,
    whisper_model: str,
    mistral_model: str,
):
    if config:
        with open(config, "r") as f:
            _config = AudioMistralConfig(**(json.load(f)))
    else:
        _config = AudioMistralConfig(**CONFIG)

    model = AudioMistralForCausalLM(_config)
    whisper = WhisperForConditionalGeneration.from_pretrained(
        f"openai/whisper-{whisper_model}"
    )
    mistral = MistralForCausalLM.from_pretrained(
        f"mistralai/Mistral-{mistral_model}"
    )
    model.encoder.load_state_dict(whisper.model.encoder.state_dict())
    model.model.load_state_dict(mistral.model.state_dict())
    for p1, p2 in zip(model.encoder.parameters(), whisper.model.encoder.parameters()):
        assert (p1.data == p2.data).all
    for p1, p2 in zip(model.model.parameters(), mistral.model.parameters()):
        assert (p1.data == p2.data).all

    model.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        type=Path,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="dump/models/audiomistral",
        type=Path,
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--whisper-model",
        default="large-v3",
        type=str,
        help="Whisper model size to use",
    )
    parser.add_argument(
        "--mistral-model",
        default="7B-Instruct-v0.3",
        type=str,
        help="Mistral model size to use",
    )
    args = parser.parse_args()

    convert(
        config=args.config,
        output_dir=args.output_dir,
        whisper_model=args.whisper_model,
        mistral_model=args.mistral_model,
    )


if __name__ == "__main__":
    main()
