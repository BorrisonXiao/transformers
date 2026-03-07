# coding=utf-8
"""Multitask ASR+TTS model for SpeechMult5 with generic module names."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqSpectrogramOutput,
)
from ...utils import logging
from .configuration_speechmult5 import SpeechMult5Config
from .modeling_speechmult5 import (
    SpeechMult5Decoder,
    SpeechMult5Encoder,
    SpeechMult5EncoderCtcHead,
    SpeechMult5PreTrainedModel,
    SpeechMult5SpeechDecoderPostnet,
    SpeechMult5SpeechDecoderPrenet,
    SpeechMult5SpeechEncoderPrenet,
    SpeechMult5SpeechToTextOutput,
    SpeechMult5SpectrogramLoss,
    SpeechMult5TextDecoderPostnet,
    SpeechMult5TextDecoderPrenet,
    SpeechMult5TextEncoderPrenet,
    _compute_sync_info_mass_from_cross_attentions,
    _encoder_attention_mask_to_lengths,
    _prepend_prefix_attention_mask,
    _prepend_prefix_to_labels,
    _resolve_ctc_blank_token_id,
    _resolve_runtime_decoder_prefix_ids,
    _resolve_tts_decoder_prefix_ids,
    _trim_sync_positions_for_ctc,
    shift_spectrograms_right,
    shift_tokens_right,
)

logger = logging.get_logger(__name__)


class SpeechMult5ForMultiTask(SpeechMult5PreTrainedModel):
    """
    Multitask SpeechMult5 model exposing generic submodules for ASR and TTS.

    Generic submodules (stable names for selective copy/load):
      - encoder / decoder (shared transformer cores)
      - text_encoder_prenet / speech_encoder_prenet
      - text_decoder_prenet / speech_decoder_prenet
      - text_decoder_postnet / speech_decoder_postnet
      - ctc_head
    """

    _tied_weights_keys = ["text_decoder_postnet.lm_head.weight"]
    _speech_pos_sinusoidal_key = "speech_encoder_prenet.pos_sinusoidal_embed.weights"

    def __init__(self, config: SpeechMult5Config):
        super().__init__(config)

        if config.vocab_size is None:
            raise ValueError(
                "SpeechMult5ForMultiTask requires `vocab_size` in config to build text modules."
            )

        # Shared transformer cores
        self.encoder = SpeechMult5Encoder(config)
        self.decoder = SpeechMult5Decoder(config)

        # Task-specific prenets/postnets under generic names
        self.text_encoder_prenet = SpeechMult5TextEncoderPrenet(config)
        self.speech_encoder_prenet = SpeechMult5SpeechEncoderPrenet(config)
        self.text_decoder_prenet = SpeechMult5TextDecoderPrenet(config)
        self.speech_decoder_prenet = SpeechMult5SpeechDecoderPrenet(config)
        self.text_decoder_postnet = SpeechMult5TextDecoderPostnet(config)
        self.speech_decoder_postnet = SpeechMult5SpeechDecoderPostnet(config)
        self.ctc_head = SpeechMult5EncoderCtcHead(config)

        self.post_init()
        self._tie_ctc_head_if_needed()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_feature_encoder(self):
        self.speech_encoder_prenet.freeze_feature_encoder()

    def get_output_embeddings(self):
        return self.text_decoder_postnet.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.text_decoder_postnet.set_output_embeddings(new_embeddings)
        self._tie_ctc_head_if_needed()

    def get_input_embeddings(self):
        return self.text_decoder_prenet.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_decoder_prenet.set_input_embeddings(value)

    def _maybe_reconcile_speech_positional_embeddings_for_load(self, state_dict):
        key = None
        for candidate in (self._speech_pos_sinusoidal_key, f"module.{self._speech_pos_sinusoidal_key}"):
            if candidate in state_dict:
                key = candidate
                break
        if key is None:
            return state_dict

        loaded_weights = state_dict[key]
        if not torch.is_tensor(loaded_weights) or loaded_weights.ndim != 2:
            return state_dict

        pos_embed = getattr(self.speech_encoder_prenet, "pos_sinusoidal_embed", None)
        if pos_embed is None or not hasattr(pos_embed, "weights") or not hasattr(pos_embed, "make_weights"):
            return state_dict

        current_weights = pos_embed.weights
        if current_weights.ndim != 2 or loaded_weights.shape[1] != current_weights.shape[1]:
            return state_dict

        loaded_len = int(loaded_weights.shape[0])
        current_len = int(current_weights.shape[0])
        if loaded_len == current_len:
            return state_dict

        target_len = max(loaded_len, current_len)
        if current_len != target_len:
            embedding_dim = int(getattr(pos_embed, "embedding_dim", int(current_weights.shape[1])))
            padding_idx = getattr(pos_embed, "padding_idx", None)
            pos_embed.make_weights(target_len, embedding_dim, padding_idx)

        if loaded_len != target_len:
            embedding_dim = int(loaded_weights.shape[1])
            padding_idx = getattr(pos_embed, "padding_idx", None)
            resized_loaded = pos_embed.get_embedding(target_len, embedding_dim, padding_idx).to(
                dtype=loaded_weights.dtype,
                device=loaded_weights.device,
            )
            resized_loaded[:loaded_len, :] = loaded_weights
            try:
                state_dict[key] = resized_loaded
            except Exception:
                copied_state_dict = OrderedDict(state_dict)
                if hasattr(state_dict, "_metadata"):
                    copied_state_dict._metadata = state_dict._metadata
                copied_state_dict[key] = resized_loaded
                state_dict = copied_state_dict

        logger.warning(
            "Auto-resized `%s` during load_state_dict from checkpoint_len=%d, model_len=%d to target_len=%d.",
            key,
            loaded_len,
            current_len,
            target_len,
        )
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        state_dict = self._maybe_reconcile_speech_positional_embeddings_for_load(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _tie_ctc_head_if_needed(self):
        if not getattr(self.config, "ctc_share_decoder_embed", False):
            return
        ctc_vocab_size = (
            int(self.config.ctc_vocab_size)
            if getattr(self.config, "ctc_vocab_size", None) is not None
            else int(self.config.vocab_size)
        )
        if ctc_vocab_size != int(self.config.vocab_size):
            raise ValueError(
                "`ctc_share_decoder_embed=True` requires `ctc_vocab_size == vocab_size`."
            )
        decoder_embed = self.text_decoder_prenet.get_input_embeddings().weight
        if self.ctc_head.proj.weight.shape != decoder_embed.shape:
            raise ValueError(
                "CTC head and decoder embedding shapes do not match for tying: "
                f"{tuple(self.ctc_head.proj.weight.shape)} vs {tuple(decoder_embed.shape)}"
            )
        self.ctc_head.proj.weight = decoder_embed

    def _alignment_scope_enabled(self, task: str) -> bool:
        weight = float(getattr(self.config, "alignment_loss_weight", 0.0) or 0.0)
        if weight <= 0.0:
            return False

        scope = str(getattr(self.config, "alignment_loss_scope", "none") or "none").strip().lower()
        if scope not in {"none", "asr", "tts", "both"}:
            raise ValueError(
                "Unsupported alignment_loss_scope: "
                f"{scope!r}. Expected one of 'none', 'asr', 'tts', or 'both'."
            )
        if scope == "none":
            return False
        if scope == "both":
            return task in {"asr", "tts"}
        return scope == task

    def _extract_sync_states(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        sync_len = int(getattr(self.config, "sync_matrix_len", 0) or 0)
        if sync_len <= 0:
            raise ValueError(
                "Alignment loss requires `config.sync_matrix_len > 0` to extract the encoder sync slice."
            )
        if hidden_states.shape[1] < sync_len:
            raise ValueError(
                "Encoder hidden state is shorter than `config.sync_matrix_len`: "
                f"{hidden_states.shape[1]} < {sync_len}."
            )
        return hidden_states[:, :sync_len, :]

    def _compute_alignment_loss(
        self,
        text_sync: torch.FloatTensor,
        speech_sync: torch.FloatTensor,
        loss_type: str,
    ) -> torch.Tensor:
        if text_sync.shape != speech_sync.shape:
            raise ValueError(
                "Alignment loss requires matching text/speech sync shapes, got "
                f"{tuple(text_sync.shape)} vs {tuple(speech_sync.shape)}."
            )

        loss_type = str(loss_type or "mse").strip().lower()
        text_sync_f = text_sync.float()
        speech_sync_f = speech_sync.float()
        if loss_type == "mse":
            return F.mse_loss(text_sync_f, speech_sync_f)
        if loss_type == "smooth_l1":
            return F.smooth_l1_loss(text_sync_f, speech_sync_f, beta=1.0)
        if loss_type == "cosine":
            cosine = F.cosine_similarity(text_sync_f, speech_sync_f, dim=-1, eps=1e-8)
            return (1.0 - cosine).mean()
        if loss_type == "cka":
            text_sync_f = text_sync_f - text_sync_f.mean(dim=1, keepdim=True)
            speech_sync_f = speech_sync_f - speech_sync_f.mean(dim=1, keepdim=True)

            cross = torch.matmul(text_sync_f, speech_sync_f.transpose(1, 2))
            text_gram = torch.matmul(text_sync_f, text_sync_f.transpose(1, 2))
            speech_gram = torch.matmul(speech_sync_f, speech_sync_f.transpose(1, 2))

            numerator = cross.square().sum(dim=(1, 2))
            text_norm = text_gram.square().sum(dim=(1, 2)).clamp_min(1e-12)
            speech_norm = speech_gram.square().sum(dim=(1, 2)).clamp_min(1e-12)
            cka = numerator / torch.sqrt(text_norm * speech_norm)
            return (1.0 - cka).mean()

        raise ValueError(
            "Unsupported alignment_loss_type: "
            f"{loss_type!r}. Expected one of 'mse', 'smooth_l1', 'cosine', or 'cka'."
        )

    # ---- Core helpers ----
    def _prepare_encoder_attention_mask_for_decoder_mtl(
        self,
        encoder_last_hidden_state: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor],
        *,
        encoder_modality: str,
        apply_sync_dropout: bool,
    ) -> Optional[torch.LongTensor]:
        sync_matrix_len = getattr(self.config, "sync_matrix_len", 0) or 0

        if encoder_modality == "speech":
            if attention_mask is not None:
                content_len = encoder_last_hidden_state.shape[1] - int(sync_matrix_len)
                encoder_attention_mask = self.speech_encoder_prenet._get_feature_vector_attention_mask(
                    content_len, attention_mask
                )
            else:
                encoder_attention_mask = None
        elif encoder_modality == "text":
            encoder_attention_mask = attention_mask
        else:
            raise ValueError(f"Unsupported encoder_modality: {encoder_modality}")

        if sync_matrix_len > 0 and encoder_attention_mask is not None:
            batch_size = encoder_attention_mask.shape[0]
            sync_mask = torch.ones(
                (batch_size, sync_matrix_len),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )

            if apply_sync_dropout:
                sync_matrix_dropout = float(getattr(self.config, "sync_matrix_dropout", 0.0) or 0.0)
                if self.training and sync_matrix_dropout > 0.0:
                    keep_prob = 1.0 - sync_matrix_dropout
                    dropout_mask = torch.rand((batch_size, 1), device=sync_mask.device) < keep_prob
                    sync_mask *= dropout_mask.to(sync_mask.dtype)
                self._last_sync_keep_rate = sync_mask[:, 0].float().mean().detach().item()

            encoder_attention_mask = torch.cat([sync_mask, encoder_attention_mask], dim=1)
        elif apply_sync_dropout:
            self._last_sync_keep_rate = None

        return encoder_attention_mask

    def _encode_text(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutput:
        hidden_states = self.text_encoder_prenet(input_ids)
        return self.encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True if return_dict is None else return_dict,
            modality_id=0,
        )

    def _encode_speech(
        self,
        input_values: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutput:
        hidden_states, encoder_input_mask = self.speech_encoder_prenet(input_values, attention_mask)
        return self.encoder(
            hidden_states=hidden_states,
            attention_mask=encoder_input_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True if return_dict is None else return_dict,
            modality_id=1,
        )

    def _decode_text(
        self,
        decoder_input_ids: torch.LongTensor,
        decoder_attention_mask: Optional[torch.LongTensor],
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: Optional[torch.LongTensor],
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        decoder_hidden_states, decoder_attention_mask = self.text_decoder_prenet(
            decoder_input_ids, decoder_attention_mask, past_key_values
        )
        return self.decoder(
            hidden_states=decoder_hidden_states,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True if return_dict is None else return_dict,
        )

    def _decode_speech(
        self,
        decoder_input_values: torch.FloatTensor,
        decoder_attention_mask: Optional[torch.LongTensor],
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: Optional[torch.LongTensor],
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        decoder_hidden_states = self.speech_decoder_prenet(
            decoder_input_values,
            speaker_embeddings,
            decoder_inputs_embeds=decoder_inputs_embeds,
        )
        return self.decoder(
            hidden_states=decoder_hidden_states,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True if return_dict is None else return_dict,
        )

    def _compute_ctc_logits_and_lengths(
        self,
        encoder_last_hidden_state: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor],
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        encoder_attention_mask = self._prepare_encoder_attention_mask_for_decoder_mtl(
            encoder_last_hidden_state,
            attention_mask,
            encoder_modality="speech",
            apply_sync_dropout=False,
        )
        ctc_hidden_states, ctc_attention_mask = _trim_sync_positions_for_ctc(
            self.config, encoder_last_hidden_state, encoder_attention_mask
        )
        ctc_logits = self.ctc_head(ctc_hidden_states)
        ctc_input_lengths = _encoder_attention_mask_to_lengths(
            ctc_hidden_states, ctc_attention_mask
        )
        return ctc_logits, ctc_input_lengths

    def _compute_content_encoder_attention_mask(
        self,
        encoder_last_hidden_state: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor],
        *,
        encoder_modality: str,
    ) -> Optional[torch.LongTensor]:
        encoder_attention_mask = self._prepare_encoder_attention_mask_for_decoder_mtl(
            encoder_last_hidden_state,
            attention_mask,
            encoder_modality=encoder_modality,
            apply_sync_dropout=False,
        )
        if encoder_modality == "speech":
            _, content_attention_mask = _trim_sync_positions_for_ctc(
                self.config, encoder_last_hidden_state, encoder_attention_mask
            )
            return content_attention_mask
        sync_len = int(getattr(self.config, "sync_matrix_len", 0) or 0)
        if encoder_attention_mask is None:
            return None
        return encoder_attention_mask[:, sync_len:]

    def get_normalized_probs_for_ctc(
        self, ctc_logits: torch.FloatTensor, log_probs: bool = True
    ) -> torch.FloatTensor:
        if log_probs:
            return F.log_softmax(ctc_logits.float(), dim=-1)
        return F.softmax(ctc_logits.float(), dim=-1)

    def ctc_greedy_decode(
        self,
        ctc_logits: torch.FloatTensor,
        input_lengths: torch.LongTensor,
        *,
        ctc_blank_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        # Reuse ASR-class-free path only if needed by future callers. Training recipe uses standard ASR model for inference.
        blank_id = _resolve_ctc_blank_token_id(self.config, ctc_blank_token_id=ctc_blank_token_id)
        pred = ctc_logits.argmax(dim=-1)
        pad_id = int(self.config.pad_token_id)
        eos_id = int(self.config.eos_token_id) if self.config.eos_token_id is not None else None
        out = torch.full((pred.size(0), pred.size(1) + 1), pad_id, dtype=torch.long, device=pred.device)
        for b in range(pred.size(0)):
            tlen = int(input_lengths[b].item())
            seq = pred[b, :tlen].tolist()
            collapsed = []
            prev = None
            for tok in seq:
                if tok == blank_id:
                    prev = tok
                    continue
                if prev is not None and tok == prev:
                    continue
                collapsed.append(int(tok))
                prev = tok
            if eos_id is not None:
                collapsed.append(eos_id)
            n = min(len(collapsed), out.size(1))
            if n > 0:
                out[b, :n] = torch.tensor(collapsed[:n], device=out.device)
        return out

    # ---- Forward dispatch ----
    def forward(self, task: Optional[str] = None, **kwargs):
        if task is None:
            raise ValueError("`task` must be provided to SpeechMult5ForMultiTask.forward (expected 'asr' or 'tts').")
        task = str(task).lower()
        if task == "asr":
            return self.forward_asr(**kwargs)
        if task == "tts":
            return self.forward_tts(**kwargs)
        raise ValueError(f"Unsupported task: {task}")

    def forward_asr(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        decoder_prefix_ids: Optional[Union[List[int], torch.LongTensor]] = None,
        ctc_labels: Optional[torch.LongTensor] = None,
        ce_weight: Optional[float] = None,
        ctc_weight: Optional[float] = None,
        output_ctc_logits: Optional[bool] = None,
    ) -> Union[Tuple, SpeechMult5SpeechToTextOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        ce_weight = 1.0 if ce_weight is None else float(ce_weight)
        ctc_weight = 0.0 if ctc_weight is None else float(ctc_weight)
        want_ctc_logits = bool(output_ctc_logits) or ctc_weight > 0.0

        labels_for_loss = labels
        labels_for_ctc = ctc_labels if ctc_labels is not None else labels
        if labels is not None and decoder_input_ids is None:
            prefix_ids = _resolve_runtime_decoder_prefix_ids(
                self.config,
                decoder_prefix_ids=decoder_prefix_ids,
                batch_size=labels.shape[0],
                device=labels.device,
            )
            if prefix_ids is not None:
                labels_for_loss = _prepend_prefix_to_labels(labels, prefix_ids)
                decoder_attention_mask = _prepend_prefix_attention_mask(
                    decoder_attention_mask,
                    prefix_length=prefix_ids.shape[1],
                    batch_size=labels.shape[0],
                    device=labels.device,
                )
            decoder_input_ids = shift_tokens_right(
                labels_for_loss,
                self.config.pad_token_id,
                self.config.decoder_start_token_id,
            )

        if decoder_input_ids is None and (ctc_weight > 0.0 or want_ctc_logits):
            batch_size = None
            device = None
            for tensor in (input_values, attention_mask):
                if tensor is not None:
                    batch_size = tensor.shape[0]
                    device = tensor.device
                    break
            if batch_size is None and encoder_outputs is not None:
                enc0 = encoder_outputs[0] if isinstance(encoder_outputs, tuple) else encoder_outputs.last_hidden_state
                batch_size = enc0.shape[0]
                device = enc0.device
            if batch_size is None:
                raise ValueError("Unable to infer batch size for CTC-only ASR forward.")
            decoder_input_ids = torch.full(
                (batch_size, 1),
                int(self.config.decoder_start_token_id),
                dtype=torch.long,
                device=device,
            )

        if encoder_outputs is None:
            encoder_outputs = self._encode_speech(
                input_values=input_values,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        elif not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        encoder_attention_mask = self._prepare_encoder_attention_mask_for_decoder_mtl(
            encoder_outputs.last_hidden_state,
            attention_mask,
            encoder_modality="speech",
            apply_sync_dropout=True,
        )

        decoder_outputs = self._decode_text(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=encoder_attention_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        logits = self.text_decoder_postnet(decoder_outputs.last_hidden_state)

        ctc_logits = None
        ctc_input_lengths = None
        if want_ctc_logits:
            ctc_logits, ctc_input_lengths = self._compute_ctc_logits_and_lengths(
                encoder_outputs.last_hidden_state,
                attention_mask,
            )

        loss = None
        loss_dict = None
        ce_loss = None
        ctc_loss = None
        alignment_loss = None
        alignment_loss_scaled = None
        if labels is not None and ce_weight > 0.0:
            loss_fct = CrossEntropyLoss()
            ce_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels_for_loss.view(-1))

        if ctc_weight > 0.0:
            if labels_for_ctc is None:
                raise ValueError("`ctc_weight > 0` requires `labels` or `ctc_labels`.")
            if ctc_logits is None or ctc_input_lengths is None:
                ctc_logits, ctc_input_lengths = self._compute_ctc_logits_and_lengths(
                    encoder_outputs.last_hidden_state,
                    attention_mask,
                )
            blank_id = _resolve_ctc_blank_token_id(self.config)
            ctc_targets_source = labels_for_ctc
            ctc_target_mask = ctc_targets_source.ne(-100)
            if self.config.eos_token_id is not None:
                ctc_target_mask = ctc_target_mask & ctc_targets_source.ne(self.config.eos_token_id)
            target_lengths = ctc_target_mask.long().sum(dim=-1)
            targets_flat = ctc_targets_source.masked_select(ctc_target_mask)
            ctc_log_probs = self.get_normalized_probs_for_ctc(ctc_logits, log_probs=True).transpose(0, 1).contiguous()
            with torch.backends.cudnn.flags(enabled=False):
                ctc_loss = F.ctc_loss(
                    ctc_log_probs,
                    targets_flat,
                    ctc_input_lengths.long(),
                    target_lengths.long(),
                    blank=blank_id,
                    reduction="mean",
                    zero_infinity=bool(getattr(self.config, "ctc_zero_infinity", True)),
                )

        if ce_loss is not None and ctc_loss is not None:
            loss = (ce_weight * ce_loss) + (ctc_weight * ctc_loss)
        elif ce_loss is not None:
            loss = ce_weight * ce_loss
        elif ctc_loss is not None:
            loss = ctc_weight * ctc_loss

        if (labels is not None or ctc_labels is not None) and ce_weight <= 0.0 and ctc_weight <= 0.0:
            raise ValueError("At least one of `ce_weight` or `ctc_weight` must be > 0 when labels are provided.")

        if self._alignment_scope_enabled("asr") and labels is not None:
            alignment_text_attention_mask = labels.ne(-100).long()
            alignment_text_input_ids = labels.masked_fill(labels.eq(-100), int(self.config.pad_token_id))
            alignment_text_outputs = self._encode_text(
                input_ids=alignment_text_input_ids,
                attention_mask=alignment_text_attention_mask,
                head_mask=head_mask,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            alignment_loss = self._compute_alignment_loss(
                text_sync=self._extract_sync_states(alignment_text_outputs.last_hidden_state),
                speech_sync=self._extract_sync_states(encoder_outputs.last_hidden_state),
                loss_type=getattr(self.config, "alignment_loss_type", "mse"),
            )
            alignment_weight = float(getattr(self.config, "alignment_loss_weight", 0.0) or 0.0)
            alignment_loss_scaled = alignment_weight * alignment_loss
            loss = loss + alignment_loss_scaled if loss is not None else alignment_loss_scaled

        if ce_loss is not None or ctc_loss is not None or alignment_loss is not None:
            loss_dict = {}
            if ce_loss is not None:
                loss_dict["ce_loss"] = float(ce_loss.detach().item())
            if ctc_loss is not None:
                loss_dict["ctc_loss"] = float(ctc_loss.detach().item())
            if alignment_loss is not None:
                loss_dict["alignment_loss"] = float(alignment_loss.detach().item())
            if alignment_loss_scaled is not None:
                loss_dict["alignment_loss_scaled"] = float(alignment_loss_scaled.detach().item())
            if decoder_outputs.cross_attentions is not None and labels_for_loss is not None:
                try:
                    content_encoder_attention_mask = self._compute_content_encoder_attention_mask(
                        encoder_outputs.last_hidden_state,
                        attention_mask,
                        encoder_modality="speech",
                    )
                    output_token_mask = labels_for_loss.ne(-100)
                    sync_mass, info_mass = _compute_sync_info_mass_from_cross_attentions(
                        cross_attentions=decoder_outputs.cross_attentions,
                        encoder_attention_mask_without_sync=content_encoder_attention_mask,
                        output_token_mask=output_token_mask,
                    )
                    loss_dict["sync_mass"] = float(sync_mass.detach().item())
                    loss_dict["info_mass"] = float(info_mass.detach().item())
                except Exception:
                    pass
            if hasattr(self, "_last_sync_keep_rate") and getattr(self, "_last_sync_keep_rate", None) is not None:
                loss_dict["sync_keep_rate"] = float(self._last_sync_keep_rate)
            if loss is not None:
                loss_dict["combined_loss"] = float(loss.detach().item())
            self._last_loss_dict = loss_dict

        if not return_dict:
            output = (logits,)
            if ctc_logits is not None:
                output = output + (ctc_logits,)
            return ((loss,) + output)

        return SpeechMult5SpeechToTextOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            ctc_logits=ctc_logits,
            ce_loss=ce_loss,
            ctc_loss=ctc_loss,
            encoder_ctc_lengths=ctc_input_lengths,
        )

    def forward_tts(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_values: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        stop_labels: Optional[torch.Tensor] = None,
        special_token_ids: Optional[torch.LongTensor] = None,
        decoder_prefix_ids: Optional[Union[List[int], torch.LongTensor]] = None,
        speech_input_values: Optional[torch.FloatTensor] = None,
        speech_attention_mask: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Seq2SeqSpectrogramOutput]:
        del stop_labels  # current loss reconstructs stop labels from padding mask

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if labels is not None:
            if decoder_input_values is None:
                decoder_input_values, decoder_attention_mask = shift_spectrograms_right(
                    labels, self.config.reduction_factor, decoder_attention_mask
                )
            if self.config.use_guided_attention_loss:
                output_attentions = True

        if output_attentions is None:
            output_attentions = self.config.output_attentions

        batch_size = None
        device = None
        for tensor in (input_ids, labels, decoder_input_values, speaker_embeddings):
            if tensor is not None:
                batch_size = tensor.shape[0]
                device = tensor.device
                break
        if batch_size is not None:
            special_token_ids = _resolve_tts_decoder_prefix_ids(
                self.config,
                decoder_prefix_ids=decoder_prefix_ids,
                special_token_ids=special_token_ids,
                batch_size=batch_size,
                device=device,
            )

        decoder_inputs_embeds = None
        if special_token_ids is not None:
            bsz = special_token_ids.shape[0]
            start_tokens = torch.full(
                (bsz, 1),
                self.config.decoder_start_token_id,
                dtype=special_token_ids.dtype,
                device=special_token_ids.device,
            )
            decoder_input_ids = torch.cat([start_tokens, special_token_ids], dim=1)
            inputs_embeds = self.text_decoder_prenet.get_input_embeddings()(decoder_input_ids)
            if self.config.scale_embedding:
                inputs_embeds = inputs_embeds * (self.config.hidden_size ** 0.5)
            positions = self.text_decoder_prenet.embed_positions(decoder_input_ids, 0)
            decoder_inputs_embeds = self.text_decoder_prenet.dropout(inputs_embeds + positions)

            if decoder_attention_mask is not None:
                prefix_mask = torch.ones(
                    (bsz, decoder_input_ids.shape[1]),
                    dtype=decoder_attention_mask.dtype,
                    device=decoder_attention_mask.device,
                )
                decoder_attention_mask = torch.cat([prefix_mask, decoder_attention_mask], dim=1)

        if encoder_outputs is None:
            encoder_outputs = self._encode_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        elif not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        encoder_attention_mask = self._prepare_encoder_attention_mask_for_decoder_mtl(
            encoder_outputs.last_hidden_state,
            attention_mask,
            encoder_modality="text",
            apply_sync_dropout=True,
        )

        decoder_outputs = self._decode_speech(
            decoder_input_values=decoder_input_values,
            decoder_attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=encoder_attention_mask,
            speaker_embeddings=speaker_embeddings,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = decoder_outputs.last_hidden_state
        special_logits = None
        if special_token_ids is not None:
            num_special = special_token_ids.shape[1] + 1
            special_states = hidden_states[:, : num_special - 1]
            speech_states = hidden_states[:, num_special:]
            embed_weight = self.text_encoder_prenet.get_input_embeddings().weight
            special_logits = F.linear(special_states, embed_weight)
        else:
            speech_states = hidden_states

        outputs_before_postnet, outputs_after_postnet, logits = self.speech_decoder_postnet(speech_states)

        loss = None
        loss_dict = {}
        alignment_loss = None
        alignment_loss_scaled = None
        if labels is not None:
            criterion = SpeechMult5SpectrogramLoss(self.config)
            (
                loss,
                l1_loss,
                bce_loss,
                ce_loss,
                attn_loss,
                sync_mass,
                info_mass,
                sync_balance_penalty,
                attn_text_penalty,
                attn_loss_unscaled,
            ) = criterion(
                attention_mask,
                outputs_before_postnet,
                outputs_after_postnet,
                logits,
                labels,
                decoder_outputs.cross_attentions,
                special_logits=special_logits,
                special_token_ids=special_token_ids,
            )
            if self._alignment_scope_enabled("tts"):
                if speech_input_values is None:
                    raise ValueError(
                        "TTS alignment loss requires paired `speech_input_values`. "
                        "Enable TTS alignment audio in the data pipeline/collator."
                    )
                speech_encoder_outputs = self._encode_speech(
                    input_values=speech_input_values,
                    attention_mask=speech_attention_mask,
                    head_mask=head_mask,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                alignment_loss = self._compute_alignment_loss(
                    text_sync=self._extract_sync_states(encoder_outputs.last_hidden_state),
                    speech_sync=self._extract_sync_states(speech_encoder_outputs.last_hidden_state),
                    loss_type=getattr(self.config, "alignment_loss_type", "mse"),
                )
                alignment_weight = float(getattr(self.config, "alignment_loss_weight", 0.0) or 0.0)
                alignment_loss_scaled = alignment_weight * alignment_loss
                loss = loss + alignment_loss_scaled if loss is not None else alignment_loss_scaled
            loss_dict = {
                "l1_loss": float(l1_loss.detach().item()) if torch.is_tensor(l1_loss) else float(l1_loss),
                "bce_loss": float(bce_loss.detach().item()) if torch.is_tensor(bce_loss) else float(bce_loss),
                "ce_loss": float(ce_loss.detach().item()) if torch.is_tensor(ce_loss) else float(ce_loss),
                "attn_loss": float(attn_loss.detach().item()) if torch.is_tensor(attn_loss) else float(attn_loss),
                "sync_mass": float(sync_mass.detach().item()) if torch.is_tensor(sync_mass) else float(sync_mass),
                "info_mass": float(info_mass.detach().item()) if torch.is_tensor(info_mass) else float(info_mass),
                "sync_balance_penalty": float(sync_balance_penalty.detach().item())
                if torch.is_tensor(sync_balance_penalty)
                else float(sync_balance_penalty),
                "attn_text_penalty": float(attn_text_penalty.detach().item())
                if torch.is_tensor(attn_text_penalty)
                else float(attn_text_penalty),
                "attn_loss_unscaled": float(attn_loss_unscaled.detach().item())
                if torch.is_tensor(attn_loss_unscaled)
                else float(attn_loss_unscaled),
            }
            if alignment_loss is not None:
                loss_dict["alignment_loss"] = float(alignment_loss.detach().item())
            if alignment_loss_scaled is not None:
                loss_dict["alignment_loss_scaled"] = float(alignment_loss_scaled.detach().item())
            if getattr(self, "_last_sync_keep_rate", None) is not None:
                loss_dict["sync_keep_rate"] = float(self._last_sync_keep_rate)
            if loss is not None:
                loss_dict["combined_loss"] = float(loss.detach().item())
            self._last_loss_dict = loss_dict

        if not return_dict:
            output = (outputs_after_postnet,)
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSpectrogramOutput(
            loss=loss,
            spectrogram=outputs_after_postnet,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
