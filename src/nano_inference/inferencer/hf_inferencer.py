import os
from copy import deepcopy
from typing import List, Optional, Union

import torch
from nano_inference.core.config import ModelConfig
from nano_inference.core.request import (
    FinishedReason,
    GenerateOutput,
    Request,
)
from nano_inference.core.sampling import SamplingParams
from nano_inference.envs.utils import ENV_UTILS_PICKLE_DUMP_ENABLED
from nano_inference.inferencer.base import InferencerBase
from nano_inference.inferencer.factory import register_inferencer
from nano_inference.utils.logger import logger
from nano_inference.utils.pickle_ops import dump_pickle
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@register_inferencer
class HuggingFaceInferencer(InferencerBase):
    name = "huggingface"

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = None
        self.is_vlm = False

    def load_model(self, model_config: ModelConfig) -> None:
        dtype = DTYPE_MAP.get(model_config.dtype, torch.float16)
        self.device = torch.device(model_config.device)

        logger.info(
            f"Loading HF model from {model_config.model_dir} "
            f"(dtype={model_config.dtype}, device={self.device})"
        )

        config = AutoConfig.from_pretrained(
            model_config.model_dir,
            trust_remote_code=True,
        )
        architectures = getattr(config, "architectures", []) or []
        self.is_vlm = self._is_vision_language_model(config)

        if self.is_vlm:
            self.processor = AutoProcessor.from_pretrained(
                model_config.model_dir,
                use_fast=False,
                trust_remote_code=True,
            )
            self.tokenizer = self.processor.tokenizer

            self.model = AutoModelForImageTextToText.from_pretrained(
                model_config.model_dir,
                dtype=dtype,
                trust_remote_code=True,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_config.model_dir,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_config.model_dir,
                dtype=dtype,
                trust_remote_code=True,
            )

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded: {architectures}")

    @staticmethod
    def _is_vision_language_model(config: AutoConfig) -> bool:
        return type(config) in AutoModelForImageTextToText._model_mapping

    def generate(
        self,
        request: Request,
    ) -> GenerateOutput:
        gen_inputs = request.generation_inputs

        if self.is_vlm:
            if self.processor is None:
                raise RuntimeError("Processor not loaded for VL model")

            text = self.tokenizer.decode(
                gen_inputs.prompt_token_ids, skip_special_tokens=False
            )

            inputs = self.processor(
                text=[text],
                images=gen_inputs.images,
                videos=gen_inputs.videos,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

        else:
            input_ids = torch.tensor(
                [gen_inputs.prompt_token_ids],
                dtype=torch.long,
                device=self.device,
            )
            inputs = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }

        prompt_len = inputs["input_ids"].shape[1]
        hf_kwargs = self._build_hf_kwargs(request)

        with torch.inference_mode():
            pickle_dump_flag = (
                os.environ.get(ENV_UTILS_PICKLE_DUMP_ENABLED, "").strip().lower()
                == "true"
            )
            output = self.model.generate(
                **inputs,
                **hf_kwargs,
                output_hidden_states=pickle_dump_flag,
                return_dict_in_generate=pickle_dump_flag,
                output_scores=pickle_dump_flag,
            )

        if pickle_dump_flag:
            output_ids = output.sequences
        else:
            output_ids = output

        if pickle_dump_flag:
            dump_pickle("hf-hidden-states", output.hidden_states)
            dump_pickle("hf-scores", output.scores)

        new_token_ids = output_ids[0, prompt_len:].tolist()

        finished_reason = self._determine_finished_reason(
            new_token_ids,
            request.eos_token_id,
            request.sampling_params.max_new_tokens,
        )

        return GenerateOutput(
            output_token_ids=new_token_ids,
            finished=True,
            finished_reason=finished_reason,
        )

    def _build_hf_kwargs(self, request: Request) -> dict:
        sp = request.sampling_params
        kwargs = {}
        kwargs["generation_config"] = self._build_generation_config(
            sampling_params=sp,
            eos_token_id=request.eos_token_id,
        )
        return kwargs

    def _build_generation_config(
        self,
        sampling_params: SamplingParams,
        eos_token_id: Optional[Union[int, List[int]]] = None,
    ):
        generation_config = deepcopy(self.model.generation_config)
        generation_config.max_new_tokens = sampling_params.max_new_tokens
        generation_config.repetition_penalty = sampling_params.repetition_penalty
        generation_config.pad_token_id = self._resolve_pad_token_id()

        if eos_token_id is not None:
            generation_config.eos_token_id = (
                eos_token_id if isinstance(eos_token_id, int) else eos_token_id[0]
            )

        if sampling_params.temperature == 0:
            generation_config.do_sample = False
            if hasattr(generation_config, "temperature"):
                delattr(generation_config, "temperature")
            if hasattr(generation_config, "top_p"):
                delattr(generation_config, "top_p")
            if hasattr(generation_config, "top_k"):
                delattr(generation_config, "top_k")
        else:
            generation_config.do_sample = True
            generation_config.temperature = sampling_params.temperature
            generation_config.top_p = sampling_params.top_p
            generation_config.top_k = (
                sampling_params.top_k if sampling_params.top_k > 0 else None
            )

        return generation_config

    def _resolve_pad_token_id(self) -> Optional[int]:
        if self.tokenizer is None:
            return None
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id
        if isinstance(eos_token_id, list):
            return eos_token_id[0]
        return eos_token_id

    @staticmethod
    def _determine_finished_reason(
        new_token_ids: List[int],
        eos_token_id: Union[int, List[int]],
        max_new_tokens: int,
    ) -> FinishedReason:
        if not new_token_ids:
            return FinishedReason.STOP

        eos_set = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id)

        if new_token_ids[-1] in eos_set:
            return FinishedReason.STOP

        if len(new_token_ids) >= max_new_tokens:
            return FinishedReason.LENGTH

        return FinishedReason.STOP
