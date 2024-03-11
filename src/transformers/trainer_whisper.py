#!/usr/bin/env python3
# Copyright 2023 Johns Hopkins University (Cihan Xiao)
# -*- coding: utf-8 -*-

from transformers import Seq2SeqTrainer
from typing import Dict, List, Optional
import torch
from functools import partial
from .utils import (
    is_torch_tpu_available,
)
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import time
import math
from .trainer_utils import (
    EvalPrediction,
    speed_metrics,
    has_length,
    denumpify_detensorize,
    EvalLoopOutput,
)
from .trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_numpify,
)
from .deepspeed import is_deepspeed_zero3_enabled, deepspeed_init
from .debug_utils import DebugOption
from torch import nn
from typing import Any, Dict, List, Optional, Union, Tuple
from packaging import version
from .utils import (
    is_apex_available,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)
from .modeling_utils import unwrap_model
from .models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.feature_extraction_utils import BatchFeature

logger = logging.get_logger(__name__)

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(
        SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_apex_available():
    from apex import amp

LANGS = {
    "ara": "arabic",
    "kor": "korean",
    "cmn": "chinese",
    "spa": "spanish",
    "rus": "russian",
}


def _schedule_dynamic_mtl_weight(
    current_step,
    warmup_steps,
    max_steps,
    min_weight,
    max_weight,
    loss_base,
):
    if min_weight == max_weight:
        return min_weight

    c = warmup_steps
    d = max_steps
    a = min_weight
    b = max_weight
    k = loss_base

    x = np.random.beta(min_weight, max_weight)
    return x
    
    # # Ensure x is within the range [c, d]
    # x = max(c, min(d, current_step))

    # # Normalize x to the range [0, 1]
    # normalized_x = (x - c) / (d - c)

    # return a + (b - a) * normalized_x**k


class WhisperTrainer(Seq2SeqTrainer):
    """
    Trainer for Whisper.
    The Seq2SeqTrainer is sufficient to perform monolingual (multitask) training.
    However, in order to extend to multilingual training, we need to override the evaluate()
    method so that the prompts are configured correctly, i.e. with the correct task token and the
    correct language token.
    """

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        if self.use_asr_prompt:
            inputs_asr = {}
            inputs_asr['input_features'] = inputs['input_features']
            inputs_asr['labels'] = inputs['labels_src']
            inputs_asr = BatchFeature(inputs_asr)
            inputs_asr = self._prepare_inputs(inputs_asr)

            use_asr_hyp_train = False
            promptless_train = True

            promptless_prob = .0
            if self.max_promptless_prob > 0.0:
                current_step = self.state.global_step + 1
                max_steps = self.args.max_steps
                # Linearly increase the promptless_prob from min_promptless_prob to max_promptless_prob
                # as the training step increases to max_steps
                promptless_prob = self.min_promptless_prob + \
                    (self.max_promptless_prob - self.min_promptless_prob) * \
                    (current_step - 1) / max_steps
                self.promptless_prob = promptless_prob

            # The ASR hyp generation is only applied if the max_sample_prob is greater than 0.0
            if self.max_sample_prob > 0.0:
                warmup_steps = self.args.warmup_steps
                current_step = self.state.global_step + 1
                max_steps = self.args.max_steps
                if warmup_steps > 0 and current_step > warmup_steps:
                    # Linearly increase the sample_prob from min_sample_prob to max_sample_prob
                    # as the training step increases after warmup_steps to max_steps
                    sample_prob = self.min_sample_prob + \
                        (self.max_sample_prob - self.min_sample_prob) * \
                        (current_step - warmup_steps - 1) / \
                        (max_steps - warmup_steps)
                    self.sample_prob = sample_prob
                    if torch.rand(1).item() < 1 - promptless_prob:
                        promptless_train = False
                    if not promptless_train and torch.rand(1).item() < sample_prob:
                        # Generate the ASR hypotheses and replace the ASR reference with the generated hypotheses
                        model.eval()
                        use_asr_hyp_train = True
                        
                        with torch.no_grad():
                            src_lang = self.src_lang
                            task = "transcribe"
                            asr_forced_decoder_ids = self.tokenizer.get_decoder_prompt_ids(
                                language=src_lang, task=task)
                            # Unify the torch datatype to match the model's datatype if model is an instance of DeepSpeedEngine
                            if hasattr(model, 'get_data_types'):
                                inputs['input_features'] = inputs['input_features'].to(
                                    model.get_data_types()[0])
                            # Note that the max_length is more strict than the one used in generation to avoid overflow
                            asr_hyps = model.generate(
                                inputs['input_features'], forced_decoder_ids=asr_forced_decoder_ids)
                            max_tgt_len = inputs['labels_tgt'].shape[1]
                            # Truncate the generated hypotheses so that the new supervision's does not exceed the max length
                            asr_hyps = asr_hyps[:, :440 - max_tgt_len]
                            asr_hyp_texts = self.tokenizer.batch_decode(
                                asr_hyps, skip_special_tokens=True)

                            # TODO: Debugging, remove later
                            # if torch.rand(1).item() < 0.1:
                            #     debug_f = "/home/hltcoe/cxiao/scale23/st/ft_exp/hf_whisper_medium_merged/spa/train-cts_sp/bmtl/lora/logdir/samples.txt"
                            #     with open(debug_f, "a") as fd:
                            #         # For debugging purposes, the new labels are written to a file
                            #         new_strs = asr_hyp_texts
                            #         org_strs = self.tokenizer.tokenizer.batch_decode(
                            #             inputs['labels_src'], fd)
                            #         for org, new in zip(org_strs, new_strs):
                            #             print("ORG:", file=fd)
                            #             print(org.strip(), file=fd)
                            #             print("NEW:", file=fd)
                            #             print(new.strip(), file=fd)
                            #         print("----------", file=fd)
            # If the max_sample_prob is 0.0, then the ASR hypotheses are not generated
            # and the ASR reference is used as the ST prompt
            inputs_st = {}
            inputs_st['input_features'] = inputs['input_features']

            startofprev_token = "<|startofprev|>"

            if promptless_train:
                inputs_st['labels'] = inputs['labels_tgt']
            else:
                # Add asr reference prompt to the prefix
                # i.e. <startofprev> [asr_ref] [st_ref]
                # [asr_ref] is the reference transcript for ASR without special tokens
                # [st_ref] is the reference transcript for ST with special tokens
                # Normally form the dataset, note that the <|startoftranscript|> token should be added
                # and the decoder_input_ids must be specified explicitly so that it's not shifted to the right
                # with a starting <|startoftranscript|> token
                # e.g. labels: [asr_ref] <|startoftranscript|> [st_ref] <|endoftext|>
                # e.g. decoder_input_ids: <|startofprev|> [asr_ref] <|startoftranscript|> [st_ref]
                # Note that the first <|endoftext|> token should not be ignored, otherwise the model will not
                # learn to terminate the generation.
                if not use_asr_hyp_train:
                    max_tgt_len = inputs['labels_tgt'].shape[1]
                    # Truncate the ASR transcript so that the new supervision's does not exceed the max length
                    _asr_tokens = inputs['labels_src']
                    _asr_tokens = _asr_tokens[:, :440 - max_tgt_len]
                    
                    asr_texts = self.tokenizer.tokenizer.batch_decode(
                        _asr_tokens, skip_special_tokens=True)
                else:
                    asr_texts = asr_hyp_texts
                st_ref_texts = self.tokenizer.tokenizer.batch_decode(
                    inputs['labels_tgt'], skip_special_tokens=False)
                st_ref_texts = [
                    f"<|startoftranscript|>{_text}" for _text in st_ref_texts]
                prompt_texts = [f"{startofprev_token}{asr_text}{st_ref_text}" for asr_text, st_ref_text in zip(
                    asr_texts, st_ref_texts)]
                _labels = self.tokenizer.tokenizer.batch_encode_plus(
                    prompt_texts, add_special_tokens=False).input_ids
                # labels are _labels[1:]
                labels = [{"input_ids": _label[1:]} for _label in _labels]
                # decoder_input_ids are _labels[:-1]
                decoder_input_ids = [{"input_ids": _label[:-1]}
                                    for _label in _labels]
                # Pad the labels and decoder_input_ids to the same length
                labels = self.tokenizer.tokenizer.pad(
                    labels, return_tensors="pt")
                decoder_input_ids = self.tokenizer.tokenizer.pad(
                    decoder_input_ids, return_tensors="pt")
                labels = labels["input_ids"].masked_fill(
                    labels.attention_mask.ne(1), -100)
                decoder_input_ids = decoder_input_ids["input_ids"]
                inputs_st['labels'] = labels
                inputs_st['decoder_input_ids'] = decoder_input_ids

                # Generate the decoder_loss_mask to mask out everything before the <|startoftranscript|> token (inlusive)
                # when computing the loss
                sos_id = self.tokenizer.tokenizer.convert_tokens_to_ids(
                    "<|startoftranscript|>")
                decoder_loss_mask = torch.ones_like(labels)
                for i, seq in enumerate(labels):
                    for j, label in enumerate(seq):
                        if label != sos_id:
                            decoder_loss_mask[i, j] = 0
                        else:
                            # Disallow BP on the <|startoftranscript|> token
                            decoder_loss_mask[i, j] = 0
                            break
                inputs_st['decoder_loss_mask'] = decoder_loss_mask.to(bool)
            inputs_st = BatchFeature(inputs_st)
            inputs_st = self._prepare_inputs(inputs_st)

        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(
                model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            current_step = self.state.global_step + 1
            if self.loss_warmup > 0 and current_step > self.loss_warmup:
                alpha = _schedule_dynamic_mtl_weight(
                    current_step=current_step,
                    warmup_steps=self.loss_warmup,
                    max_steps=self.args.max_steps,
                    min_weight=self.min_alpha,
                    max_weight=self.max_alpha,
                    loss_base=self.loss_base,
                )
            else:
                alpha = self.min_alpha
            if self.use_asr_prompt:
                loss_asr, outputs_asr = self.compute_loss(
                    model, inputs_asr, return_outputs=True)
                # # Re-use encoder_last_state to speed up training
                # inputs_st['encoder_outputs'] = (
                #     outputs_asr['encoder_last_hidden_state'])
                loss_st = self.compute_loss(model, inputs_st)
                loss = (1 - alpha) * loss_asr + alpha * loss_st
            else:
                loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def __init__(
        self,
        *args,
        use_asr_prompt: bool = False,
        min_promptless_prob: Optional[float] = 0.0,
        max_promptless_prob: Optional[float] = 0.0,
        min_sample_prob: Optional[float] = 0.0,
        max_sample_prob: Optional[float] = 0.0,
        src_lang: Optional[str] = None,
        eval_steps: Optional[int] = None,
        min_alpha: Optional[float] = 0.5,
        max_alpha: Optional[float] = 0.5,
        loss_warmup: Optional[int] = -1,
        loss_base: Optional[float] = 0.25,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.min_promptless_prob = min_promptless_prob
        self.max_promptless_prob = max_promptless_prob
        self.min_sample_prob = min_sample_prob
        self.max_sample_prob = max_sample_prob
        self.src_lang = src_lang
        self.use_asr_prompt = use_asr_prompt
        self.eval_steps = eval_steps
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.loss_warmup = self.args.warmup_steps if loss_warmup == -1 else loss_warmup
        self.loss_base = loss_base

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            logs["sample_prob"] = self.sample_prob if hasattr(
                self, "sample_prob") else 0.0

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    lang, mode = eval_dataset_name.split("_")
                    if "module" in model.__dict__ and type(model.module).__name__ == "PeftModel":
                        model.module.base_model.generate = partial(
                            model.module.base_model.generate, language=LANGS[lang], task="transcribe" if mode == "asr" else "translate")
                    else:
                        model.generate = partial(
                            model.generate, language=LANGS[lang], task="transcribe" if mode == "asr" else "translate")
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                        nolog=True,
                    )
                    metrics.update(dataset_metrics)
                # Sum the sacrebleu metric over all the datasets evaluated on.
                sum_sacrebleu = 0
                for key, value in metrics.items():
                    if key.endswith("_sacrebleu"):
                        sum_sacrebleu += value
                metrics["eval_sacrebleu"] = sum_sacrebleu / \
                    len(self.eval_dataset)
                self.log(metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        nolog=False,
        **gen_kwargs,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get(
                "num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        if not nolog:
            self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def _create_prompted_labels(self, inputs, asr_hyp_texts=None) -> Dict[str, torch.Tensor]:
        inputs_st = {}
        inputs_st['input_features'] = inputs['input_features']

        startofprev_token = "<|startofprev|>"

        # Add asr reference prompt to the prefix
        # i.e. <startofprev> [asr_ref] [st_ref]
        # [asr_ref] is the reference transcript for ASR without special tokens
        # [st_ref] is the reference transcript for ST with special tokens
        # Normally form the dataset, note that the <|startoftranscript|> token should be added
        # and the decoder_input_ids must be specified explicitly so that it's not shifted to the right
        # with a starting <|startoftranscript|> token
        # e.g. labels: [asr_ref] <|startoftranscript|> [st_ref] <|endoftext|>
        # e.g. decoder_input_ids: <|startofprev|> [asr_ref] <|startoftranscript|> [st_ref]
        # Note that the first <|endoftext|> token should not be ignored, otherwise the model will not
        # learn to terminate the generation.
        asr_texts = self.tokenizer.tokenizer.batch_decode(
            inputs['labels_src'], skip_special_tokens=True) if asr_hyp_texts is None else asr_hyp_texts
        st_ref_texts = self.tokenizer.tokenizer.batch_decode(
            inputs['labels_tgt'], skip_special_tokens=False)
        st_ref_texts = [
            f"<|startoftranscript|>{_text}" for _text in st_ref_texts]
        prompt_texts = [f"{startofprev_token}{asr_text}{st_ref_text}" for asr_text, st_ref_text in zip(
            asr_texts, st_ref_texts)]
        _labels = self.tokenizer.tokenizer.batch_encode_plus(
            prompt_texts, add_special_tokens=False).input_ids
        # labels are _labels[1:]
        labels = [{"input_ids": _label[1:]} for _label in _labels]
        # decoder_input_ids are _labels[:-1]
        decoder_input_ids = [{"input_ids": _label[:-1]}
                             for _label in _labels]
        # Pad the labels and decoder_input_ids to the same length
        labels = self.tokenizer.tokenizer.pad(
            labels, return_tensors="pt")
        decoder_input_ids = self.tokenizer.tokenizer.pad(
            decoder_input_ids, return_tensors="pt")
        labels = labels["input_ids"].masked_fill(
            labels.attention_mask.ne(1), -100)
        decoder_input_ids = decoder_input_ids["input_ids"]
        inputs_st['labels'] = labels
        inputs_st['decoder_input_ids'] = decoder_input_ids

        # Generate the decoder_loss_mask to mask out everything before the <|startoftranscript|> token (inlusive)
        # when computing the loss
        sos_id = self.tokenizer.tokenizer.convert_tokens_to_ids(
            "<|startoftranscript|>")
        decoder_loss_mask = torch.ones_like(labels)
        for i, seq in enumerate(labels):
            for j, label in enumerate(seq):
                if label != sos_id:
                    decoder_loss_mask[i, j] = 0
                else:
                    # Disallow BP on the <|startoftranscript|> token
                    decoder_loss_mask[i, j] = 0
                    break
        inputs_st['decoder_loss_mask'] = decoder_loss_mask.to(bool)
        inputs_st = BatchFeature(inputs_st)
        inputs_st = self._prepare_inputs(inputs_st)

        return inputs_st

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, loss_asr, logits and labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # gen_kwargs > model.generation_config > default GenerationConfig()
        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get(
                "num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get(
                "synced_gpus") is not None else default_synced_gpus
        )

        use_asr_hyp = hasattr(self, "use_asr_prompt") and self.use_asr_prompt

        if use_asr_hyp:
            has_labels = True
            inputs_asr = {}
            inputs_asr["input_features"] = inputs["input_features"]
            inputs_asr["labels"] = inputs["labels_src"]
            inputs_asr = self._prepare_inputs(inputs_asr)
        else:
            has_labels = "labels" in inputs
            inputs = self._prepare_inputs(inputs)

        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in inputs
            and "decoder_input_ids" in inputs
            and inputs["labels"].shape == inputs["decoder_input_ids"].shape
        ):
            inputs = {k: v for k, v in inputs.items() if k !=
                      "decoder_input_ids"}

        # Generate ASR hypotheses first
        if use_asr_hyp:
            self.model.generate = partial(
                self.model.generate, language=self.src_lang, task="transcribe")
            asr_hyps = self.model.generate(**inputs_asr, **gen_kwargs)
            self.model.generate = partial(
                self.model.generate, language=self.src_lang, task="translate")
            # TODO: As currently the only way of implementing prompting is through the use of
            # forced_decoder_ids, which is tied to the model's generate method, batched
            # generation is not supported for now.
            startofprev_id = self.tokenizer.tokenizer.convert_tokens_to_ids(
                "<|startofprev|>")
            startoftranscript_id = self.tokenizer.tokenizer.convert_tokens_to_ids(
                "<|startoftranscript|>")
            st_hyps = []
            # TODO: Formalize this
            # With a 0.5 probability, the ASR hypotheses are used as the ST prompt
            # Otherwise, the ASR reference is used as the ST prompt
            # This way, the best model should be able to do well in both tasks
            # if torch.rand(1).item() < 0.5:
            #     asr_hyp_texts = self.tokenizer.tokenizer.batch_decode(
            #         asr_hyps, skip_special_tokens=True)
            # else:
            #     asr_hyp_texts = self.tokenizer.tokenizer.batch_decode(
            #         inputs["labels_src"], skip_special_tokens=True)
            asr_hyp_texts = self.tokenizer.tokenizer.batch_decode(
                asr_hyps, skip_special_tokens=True)
            prompted_inputs = self._create_prompted_labels(
                inputs=inputs, asr_hyp_texts=asr_hyp_texts)
            for i, asr_hyp in enumerate(asr_hyp_texts):
                prompt_ids = torch.Tensor([startofprev_id] + self.tokenizer.tokenizer.encode(
                    asr_hyp, add_special_tokens=False)).long().to(self.args.device)
                inputs_st = {}
                inputs_st["input_features"] = inputs["input_features"][i].unsqueeze(
                    0)
                # Unify the torch datatype to match the model's datatype if model is an instance of DeepSpeedEngine
                if hasattr(self.model, 'get_data_types'):
                    inputs['input_features'] = inputs['input_features'].to(
                        self.model.get_data_types()[0])
                # Note that the paddings (-100) are removed as the batch size is 1
                inputs_st["labels"] = inputs["labels_tgt"][i][inputs["labels_tgt"]
                                                              [i] != -100].unsqueeze(0).to(self.args.device)
                inputs_st = self._prepare_inputs(inputs_st)
                st_hyp = self.model.generate(
                    **inputs_st, **gen_kwargs, prompt_ids=prompt_ids)
                # The generated tokens start after the first <|startoftranscript|> token
                # i.e. it contains only the translation
                # Note that the first <|startoftranscript|> token is kept
                hyp_start_pos = (st_hyp == startoftranscript_id).nonzero(
                    as_tuple=True)[-1][0]
                st_hyps.append(st_hyp[0][hyp_start_pos:])

            _generated_tokens = [{"input_ids": st_hyp} for st_hyp in st_hyps]
            # Pad the tokens to the same length with the padding token
            generated_tokens = self.tokenizer.tokenizer.pad(
                _generated_tokens, return_tensors="pt").to(self.args.device)["input_ids"]
        else:
            generated_tokens = self.model.generate(**inputs, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    if use_asr_hyp:
                        outputs_asr = self.model(**inputs_asr)

                        # TODO: Add loss_asr to the final loss for logging as well
                        outputs_st = self.model(**prompted_inputs)
                        outputs = outputs_st
                    else:
                        outputs = model(**inputs)
                if self.label_smoother is not None:
                    # TODO: Currently does not really support label smoothing under use_asr_hyp
                    loss = self.label_smoother(
                        outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(
                        outputs, dict) else outputs[0]).mean().detach()
                    if use_asr_hyp:
                        loss_asr = (outputs_asr["loss"] if isinstance(
                            outputs_asr, dict) else outputs_asr[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"] if not use_asr_hyp else inputs["labels_tgt"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(
                    labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(
                    labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return (loss, loss_asr, generated_tokens, labels) if use_asr_hyp else (loss, generated_tokens, labels)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(
            self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        # Customized for storing multiple losses
        losses_asr_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        use_asr_hyp = hasattr(self, "use_asr_prompt") and self.use_asr_prompt

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # To limit eval steps to a specific number
            if hasattr(self, "eval_steps") and self.eval_steps is not None and step >= self.eval_steps:
                break
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            if use_asr_hyp:
                loss, loss_asr, logits, labels = self.prediction_step(
                    model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            else:
                loss, logits, labels = self.prediction_step(
                    model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(
                inputs["input_ids"]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics(
                    (loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(
                    losses_host, losses, padding_index=-100)
                # Keep track of th additional ASR loss
                if use_asr_hyp:
                    losses_asr_host = loss_asr if losses_asr_host is None else nested_concat(
                        losses_asr_host, loss_asr, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(
                    inputs_decode)
                inputs_decode = self.accelerator.gather_for_metrics(
                    (inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.accelerator.gather_for_metrics((logits))
                preds_host = logits if preds_host is None else nested_concat(
                    preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self.accelerator.gather_for_metrics((labels))
                labels_host = labels if labels_host is None else nested_concat(
                    labels_host, labels, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate(
                        (all_losses, losses), axis=0)
                if losses_asr_host is not None:
                    losses_asr = nested_numpify(losses_asr_host)
                    all_losses_asr = (
                        losses_asr if all_losses_asr is None else np.concatenate(
                            (all_losses_asr, losses_asr), axis=0)
                    )
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(
                        all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(
                            all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None
                if use_asr_hyp:
                    losses_asr_host = None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            all_losses = nested_numpify(losses_host)
        if preds_host is not None:
            all_preds = nested_numpify(preds_host)
        if inputs_host is not None:
            all_inputs = nested_numpify(inputs_host)
        if labels_host is not None:
            all_labels = nested_numpify(labels_host)
        if use_asr_hyp:
            if losses_asr_host is not None:
                all_losses_asr = nested_numpify(losses_asr_host)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds,
                                   label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(
                    predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if use_asr_hyp:
            if all_losses_asr is not None:
                metrics[f"{metric_key_prefix}_loss_asr"] = all_losses_asr.mean(
                ).item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
