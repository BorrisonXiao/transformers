#!/usr/bin/env python3
# Copyright 2023 Johns Hopkins University (Cihan Xiao)
# -*- coding: utf-8 -*-

import os
from transformers import Seq2SeqTrainer
from typing import Dict, List, Optional
import torch
from functools import partial
from .utils import (
    is_torch_tpu_available,
)
from torch.utils.data.dataset import Dataset
import time
import math
from .trainer_utils import (
    EvalPrediction,
    speed_metrics,
)
from .debug_utils import DebugOption
from torch import nn
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from packaging import version
from .utils import (
    is_apex_available,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)
from .modeling_utils import unwrap_model
from .models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from .modeling_utils import PreTrainedModel
from .training_args import TrainingArguments
from .tokenization_utils_base import PreTrainedTokenizerBase
from .data.data_collator import DataCollator
from .trainer_callback import (
    TrainerCallback,
)

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
                if torch.rand(1).item() < sample_prob:
                    # Generate the ASR hypotheses and replace the ASR reference with the generated hypotheses
                    model.eval()
                    with torch.no_grad():
                        src_lang = self.src_lang
                        task = "transcribe"
                        asr_forced_decoder_ids = self.tokenizer.get_decoder_prompt_ids(
                            language=src_lang, task=task)
                        # print(model)
                        # print(dir(model))
                        # Unify the torch datatype to match the model's datatype if model is an instance of DeepSpeedEngine
                        if hasattr(model, 'get_data_types'):
                            inputs['input_features'] = inputs['input_features'].to(
                                model.get_data_types()[0])
                        # Note that the max_length is more strict than the one used in generation to avoid overflow
                        asr_hyps = model.generate(
                            inputs['input_features'], forced_decoder_ids=asr_forced_decoder_ids)
                        # Create a new batch of inputs, replacing the ASR reference with the generated hypotheses
                        new_inputs = {}
                        label_features = []
                        for key, value in inputs.items():
                            if key == 'labels':
                                for i, hyp in enumerate(asr_hyps):
                                    # The new labels are constructed in the manner below:
                                    # [prefixes, asr_hyp, <|startoftranslation|>, st_ref, <|endoftext|>]
                                    # The key operation is to replace the ASR reference with the generated hypotheses
                                    # and rearrange the paddings.
                                    # Also note that the paddings in the ASR hypotheses are not removed.
                                    ref = value[i]
                                    # Remove the leading <|startoftranscript|>[langid]<|transcribe|><|notimestamps|> tokens
                                    _hyp = hyp[4:]
                                    _hyp = _hyp[_hyp !=
                                                self.tokenizer.tokenizer.pad_token_id]
                                    startoftranslation_id = self.tokenizer.tokenizer.convert_tokens_to_ids(
                                        "<|startoftranslation|>")
                                    startoftranslation_pos = torch.where(
                                        value[i] == startoftranslation_id)[0][0]
                                    st_ref = ref[startoftranslation_pos:]
                                    st_ref = st_ref[st_ref != -100]
                                    label_features.append({'input_ids': torch.cat(
                                        [ref[:3], _hyp[:440 - len(st_ref)], st_ref])})
                                labels_batch = self.tokenizer.tokenizer.pad(
                                    label_features, return_tensors="pt")
                                new_inputs[key] = labels_batch["input_ids"].masked_fill(
                                    labels_batch.attention_mask.ne(1), -100)

                                # TODO: Debugging, remove later
                                debug_f = "/home/hltcoe/cxiao/scale23/st/ft_exp/hf_whisper_medium_merged/spa/train-cts_sp/bmtl/lora/logdir/samples.txt"
                                with open(debug_f, "a") as fd:
                                    # For debugging purposes, the new labels are written to a file
                                    new_strs = self.tokenizer.tokenizer.batch_decode(
                                        labels_batch["input_ids"], fd)
                                    org_strs = self.tokenizer.tokenizer.batch_decode(
                                        value, fd)
                                    for org, new in zip(org_strs, new_strs):
                                        print("ORG:", file=fd)
                                        print(org.strip(), file=fd)
                                        print("NEW:", file=fd)
                                        print(new.strip(), file=fd)
                                    print("----------", file=fd)
                            else:
                                new_inputs[key] = value
                        inputs = new_inputs
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(
                model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
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

            if hasattr(self, "mask_asr_hyp") and self.mask_asr_hyp == True:
                decoder_loss_mask = torch.ones_like(inputs['labels'])
                for i, seq in enumerate(inputs['labels']):
                    for j, label in enumerate(seq):
                        # Allow loss to be computed for the first 3 tokens (lid and other prefixes)
                        if j < 3:
                            continue
                        startoftranslation_id = self.tokenizer.tokenizer.convert_tokens_to_ids(
                            "<|startoftranslation|>")
                        if label != startoftranslation_id:
                            decoder_loss_mask[i, j] = 0
                        else:
                            break
                inputs['decoder_loss_mask'] = decoder_loss_mask.to(bool)
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
        mask_asr_hyp: Optional[bool] = False,
        min_sample_prob: Optional[float] = 0.0,
        max_sample_prob: Optional[float] = 0.0,
        src_lang: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mask_asr_hyp = mask_asr_hyp
        self.min_sample_prob = min_sample_prob
        self.max_sample_prob = max_sample_prob
        self.src_lang = src_lang

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
