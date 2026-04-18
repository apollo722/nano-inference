from __future__ import annotations

from typing import List, Optional, Union

import torch
from nano_inference.core.config import ModelConfig
from nano_inference.core.request import FinishedReason, GenerateOutput, Request
from nano_inference.core.sampling import SamplingParams
from nano_inference.inferencer.base import InferencerBase
from nano_inference.inferencer.factory import register_inferencer
from nano_inference.model_loader import LoadReport, load_hf_config, select_loader
from nano_inference.sampling import Sampler
from nano_inference.utils.logger import logger
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@register_inferencer
class TorchInferencer(InferencerBase):
    name = "torch"

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.load_report: Optional[LoadReport] = None
        self.sampler = Sampler()

    def load_model(self, model_config: ModelConfig) -> None:
        dtype = DTYPE_MAP.get(model_config.dtype, torch.float16)
        self.device = torch.device(model_config.device)

        logger.info(
            f"Loading torch model from {model_config.model_dir} "
            f"(dtype={model_config.dtype}, device={self.device})"
        )

        config = AutoConfig.from_pretrained(
            model_config.model_dir,
            trust_remote_code=True,
        )

        architectures = getattr(config, "architectures", []) or []

        # Check if this is a VL model - we don't support VL yet
        if self._is_vision_language_model(config):
            raise NotImplementedError(
                f"Vision-language models are not yet supported by TorchInferencer. "
                f"Detected model architecture: {architectures}. "
                f"Please use HuggingFaceInferencer for VL models, or use a text-only LLM with TorchInferencer."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_dir,
            trust_remote_code=True,
        )

        hf_config = load_hf_config(model_config.model_dir)
        loader = select_loader(hf_config)
        decoder_config = loader.to_decoder_config(
            hf_config,
            runtime_config=model_config,
        )

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_config.model_dir,
            dtype=dtype,
            trust_remote_code=True,
        )

        remapped_state_dict = loader.remap_state_dict(
            hf_model.state_dict(),
            decoder_config,
        )

        self.model = loader.build_model(decoder_config)
        incompatible = self.model.load_state_dict(remapped_state_dict, strict=False)
        self.load_report = LoadReport(
            loaded_keys=sorted(remapped_state_dict.keys()),
            missing_keys=list(incompatible.missing_keys),
            unexpected_keys=list(incompatible.unexpected_keys),
        )

        self.model.to(device=self.device, dtype=dtype)
        self.model.eval()

        logger.info(f"Model loaded: {architectures}")

    def generate(
        self,
        request: Request,
    ) -> GenerateOutput:
        if self.model is None or self.device is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        gen_inputs = request.generation_inputs
        generated_ids = list(gen_inputs.prompt_token_ids)
        new_token_ids: List[int] = []

        with torch.inference_mode():
            for _ in range(request.sampling_params.max_new_tokens):
                input_ids = torch.tensor(
                    [generated_ids],
                    dtype=torch.long,
                    device=self.device,
                )

                logits = self.model(input_ids=input_ids)
                next_token_logits = logits[0, -1, :].float()

                next_token_id = self.sampler.select(
                    logits=next_token_logits,
                    generated_ids=generated_ids,
                    sampling_params=request.sampling_params,
                )

                generated_ids.append(next_token_id)
                new_token_ids.append(next_token_id)

                if self._is_eos_token(next_token_id, request.eos_token_id):
                    return GenerateOutput(
                        output_token_ids=new_token_ids,
                        finished=True,
                        finished_reason=FinishedReason.STOP,
                    )

        return GenerateOutput(
            output_token_ids=new_token_ids,
            finished=True,
            finished_reason=FinishedReason.LENGTH,
        )

    @staticmethod
    def _is_eos_token(
        token_id: int,
        eos_token_id: Union[int, List[int]],
    ) -> bool:
        if isinstance(eos_token_id, int):
            return token_id == eos_token_id
        return token_id in eos_token_id

    @staticmethod
    def _is_vision_language_model(config: AutoConfig) -> bool:
        # Check for common VLM model attributes
        if hasattr(config, "vision_config"):
            return True
        architectures = getattr(config, "architectures", []) or []
        for arch in architectures:
            if "Vision" in arch or "VL" in arch or "ImageText" in arch:
                return True
        return False
