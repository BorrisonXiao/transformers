#!/usr/bin/env python3
# Copyright 2023 Johns Hopkins University (Cihan Xiao)
# -*- coding: utf-8 -*-

from transformers import Seq2SeqTrainer
from typing import Dict
import torch
from functools import partial
from .utils import (
    is_torch_tpu_available,
)

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

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
                    model.generate = partial(model.generate, language=LANGS[lang], task="transcribe" if mode == "asr" else "translate")
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
                # Sum the sacrebleu metric over all the datasets evaluated on.
                sum_sacrebleu = 0
                for key, value in metrics.items():
                    if key.endswith("_sacrebleu"):
                        sum_sacrebleu += value
                metrics["eval_sacrebleu"] = sum_sacrebleu / len(self.eval_dataset)
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
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)