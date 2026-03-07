# coding=utf-8
# Copyright 2023 The Facebook Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization class for SpeechMult5."""

import json
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm

from ...processing_utils import ProcessorMixin
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
from ..wav2vec2.tokenization_wav2vec2 import Wav2Vec2CTCTokenizer
from .number_normalizer import EnglishNumberNormalizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spm_char.model"}


class SpeechMult5Tokenizer(PreTrainedTokenizer):
    """
    Construct a SpeechMult5 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The begin of sequence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        normalize (`bool`, *optional*, defaults to `False`):
            Whether to convert numeric quantities in the text to their spelt-out english counterparts.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        normalize=False,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        self.vocab_file = vocab_file
        self.normalize = normalize
        self._normalizer = None

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            normalize=normalize,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        normalize = kwargs.pop("normalize", self.normalize)
        if is_split_into_words:
            text = " " + text
        if normalize:
            text = self.normalizer(text)
        return (text, kwargs)

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    @property
    def normalizer(self):
        if self._normalizer is None:
            self._normalizer = EnglishNumberNormalizer()
        return self._normalizer

    @normalizer.setter
    def normalizer(self, value):
        self._normalizer = value

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    # Copied from transformers.models.albert.tokenization_albert.AlbertTokenizer.convert_tokens_to_string
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        suffix_ones = [1]
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + suffix_ones
        return ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)


class SpeechMult5CharTokenizer(Wav2Vec2CTCTokenizer):
    """
    Character tokenizer used by local SpeechMult5 fine-tuning checkpoints (`vocab.json`).

    This tokenizer appends EOS in `build_inputs_with_special_tokens` to match SpeechMult5
    seq2seq usage and defaults to SentencePiece-style whitespace marker `▁`.
    """

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="▁",
        replace_word_delimiter_char=" ",
        do_lower_case=False,
        target_lang=None,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            word_delimiter_token=word_delimiter_token,
            replace_word_delimiter_char=replace_word_delimiter_char,
            do_lower_case=do_lower_case,
            target_lang=target_lang,
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        # Delegate to HF loading logic (supports added_tokens.json).
        tok = super().from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        # Some legacy configs store `|` delimiter despite SP-style vocab.
        try:
            tok.word_delimiter_token = "▁"
            tok.replace_word_delimiter_char = " "
        except Exception:
            pass
        return tok

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def _load_added_tokens_json(self, tokenizer_path: str) -> None:
        # Convenience helper used by local scripts/tests when a tokenizer is created directly.
        path = os.path.join(tokenizer_path, "added_tokens.json")
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                added_tokens = json.load(f)
            # from_pretrained already handles this, so direct-instantiation callers can opt-in.
            self.add_tokens(list(added_tokens.keys()), special_tokens=True)
        except Exception:
            return


class SpeechMult5CharProcessor(ProcessorMixin):
    """
    Lightweight processor pairing a feature extractor with `SpeechMult5CharTokenizer`.

    This mirrors `SpeechMult5Processor` behavior while avoiding strict tokenizer-type checks,
    which is useful for local character-vocab checkpoints.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "SpeechT5FeatureExtractor"
    tokenizer_class = "SpeechMult5CharTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.current_processor = self.feature_extractor

    def __call__(self, *args, **kwargs):
        audio = kwargs.pop("audio", None)
        text = kwargs.pop("text", None)
        text_target = kwargs.pop("text_target", None)
        audio_target = kwargs.pop("audio_target", None)
        sampling_rate = kwargs.pop("sampling_rate", None)

        if audio is not None and text is not None:
            raise ValueError("Cannot process both `audio` and `text` inputs.")
        if audio_target is not None and text_target is not None:
            raise ValueError("Cannot process both `audio_target` and `text_target` inputs.")
        if audio is None and audio_target is None and text is None and text_target is None:
            raise ValueError("Provide one of `audio`, `audio_target`, `text`, `text_target`.")

        if audio is not None:
            inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
        elif text is not None:
            inputs = self.tokenizer(text, **kwargs)
        else:
            inputs = None

        if audio_target is not None:
            targets = self.feature_extractor(audio_target=audio_target, *args, sampling_rate=sampling_rate, **kwargs)
            labels = targets["input_values"]
        elif text_target is not None:
            targets = self.tokenizer(text_target, **kwargs)
            labels = targets["input_ids"]
        else:
            targets = None

        if inputs is None:
            return targets
        if targets is not None:
            inputs["labels"] = labels
            decoder_attention_mask = targets.get("attention_mask")
            if decoder_attention_mask is not None:
                inputs["decoder_attention_mask"] = decoder_attention_mask
        return inputs

    def pad(self, *args, **kwargs):
        input_values = kwargs.pop("input_values", None)
        input_ids = kwargs.pop("input_ids", None)
        labels = kwargs.pop("labels", None)

        if input_values is not None and input_ids is not None:
            raise ValueError("Cannot process both `input_values` and `input_ids`.")
        if input_values is None and input_ids is None and labels is None:
            raise ValueError("Need `input_values`, `input_ids`, or `labels`.")

        if input_values is not None:
            inputs = self.feature_extractor.pad(input_values, *args, **kwargs)
        elif input_ids is not None:
            inputs = self.tokenizer.pad(input_ids, **kwargs)
        else:
            inputs = None

        if labels is not None:
            if "input_ids" in labels or (isinstance(labels, list) and labels and "input_ids" in labels[0]):
                targets = self.tokenizer.pad(labels, **kwargs)
                padded_labels = targets["input_ids"]
            else:
                feature_size_hack = self.feature_extractor.feature_size
                self.feature_extractor.feature_size = self.feature_extractor.num_mel_bins
                targets = self.feature_extractor.pad(labels, *args, **kwargs)
                self.feature_extractor.feature_size = feature_size_hack
                padded_labels = targets["input_values"]
        else:
            targets = None
            padded_labels = None

        if inputs is None:
            return targets
        if targets is not None:
            inputs["labels"] = padded_labels
            decoder_attention_mask = targets.get("attention_mask")
            if decoder_attention_mask is not None:
                inputs["decoder_attention_mask"] = decoder_attention_mask
        return inputs

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
