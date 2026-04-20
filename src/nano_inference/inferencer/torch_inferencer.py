from __future__ import annotations

from typing import List, Optional, Union

import torch
from nano_inference.core.config import ModelConfig
from nano_inference.core.context import GenerateContext
from nano_inference.core.request import (
    FinishedReason,
    GenerateOutput,
    GenerationStage,
    Request,
)
from nano_inference.core.sampling import SamplingParams
from nano_inference.inferencer.base import InferencerBase
from nano_inference.inferencer.factory import register_inferencer
from nano_inference.model_loader import LoadReport, load_hf_config, select_loader
from nano_inference.sampling import Sampler
from nano_inference.utils.logger import logger
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
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
        self.processor = None  # set for VLM models (HF AutoProcessor)
        self.device = None
        self.load_report: Optional[LoadReport] = None
        self.sampler = Sampler()
        self._is_vlm: bool = False

    @property
    def is_vlm(self) -> bool:
        return self._is_vlm

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

        self._is_vlm = self._is_vision_language_model(config)

        if self._is_vlm:
            # Load processor (contains both tokenizer and image_processor)
            processor = AutoProcessor.from_pretrained(
                model_config.model_dir,
                trust_remote_code=True,
            )
            self.processor = processor
            self.tokenizer = processor.tokenizer
            hf_model = AutoModel.from_pretrained(
                model_config.model_dir,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_config.model_dir,
                trust_remote_code=True,
            )
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_config.model_dir,
                torch_dtype=dtype,
                trust_remote_code=True,
            )

        hf_config = load_hf_config(model_config.model_dir)
        loader = select_loader(hf_config)
        decoder_config = loader.to_decoder_config(
            hf_config,
            runtime_config=model_config,
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

        # For VLM: wire the image_processor into the model for prefill
        if self._is_vlm and hasattr(processor, "image_processor"):
            self.model.model.image_processor = processor.image_processor

        architectures = getattr(config, "architectures", []) or []
        logger.info(f"Model loaded: {architectures}")

    def generate(
        self,
        request: Request,
    ) -> GenerateOutput:
        if self.model is None or self.device is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        # Build a temporary GenerateQuery to reuse the step-based approach
        from nano_inference.core.request import GenerateQuery

        query = GenerateQuery.from_request(request)
        from nano_inference.engine.context_builder import GenerateContextBuilder
        from nano_inference.kv_cache import PagedKVCacheAllocator

        # For standalone generate(), we create a transient allocator
        num_kv_heads = (
            getattr(self.model.config, "num_kv_heads", None)
            or self.model.config.num_heads
        )
        head_dim = getattr(self.model.config, "head_dim", None) or (
            self.model.config.hidden_size // self.model.config.num_heads
        )
        allocator = PagedKVCacheAllocator(
            num_blocks=128,
            block_size=16,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            num_layers=self.model.config.num_layers,
            dtype=next(self.model.parameters()).dtype,
            device=str(self.device),
        )

        builder = GenerateContextBuilder(self.device)

        new_token_ids: List[int] = []
        with torch.inference_mode():
            # Initial allocation for prefill
            query.kv_cache_block = allocator.allocate(
                len(query.generation_inputs.prompt_token_ids)
                + request.sampling_params.max_new_tokens
            )

            for step_idx in range(request.sampling_params.max_new_tokens):
                logger.debug(
                    f"[TorchInferencer] Standalone generate step {step_idx}..."
                )
                context = builder.build([query])
                next_tokens = self.step(
                    context,
                    [request.sampling_params],
                    k_cache=allocator.k_cache,
                    v_cache=allocator.v_cache,
                )
                next_token_id = next_tokens[0]

                query.output_token_ids.append(next_token_id)
                new_token_ids.append(next_token_id)

                # Transition to DECODE stage after the first step
                if query.stage == GenerationStage.PREFILL:
                    query.stage = GenerationStage.DECODE

                if self._is_eos_token(next_token_id, request.eos_token_id):
                    logger.debug(f"[TorchInferencer] EOS detected at step {step_idx}")
                    allocator.free(query.kv_cache_block)
                    return GenerateOutput(
                        output_token_ids=new_token_ids,
                        finished=True,
                        finished_reason=FinishedReason.STOP,
                        full_text=self.tokenizer.decode(new_token_ids),
                    )

            logger.debug("[TorchInferencer] Max tokens reached.")
            full_text = self.tokenizer.decode(new_token_ids)
            allocator.free(query.kv_cache_block)

        return GenerateOutput(
            output_token_ids=new_token_ids,
            finished=True,
            finished_reason=FinishedReason.LENGTH,
            full_text=full_text,
        )

    def step(
        self,
        context: GenerateContext,
        all_sampling_params: List[SamplingParams],
        k_cache: torch.Tensor = None,
        v_cache: torch.Tensor = None,
    ) -> List[int]:
        """Run a single inference step for a batch.

        Phase 3: Paged Attention path.
        Phase 4: Passes VLM images on prefill.
        """
        if self.model is None or self.device is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        # Pass physical KV tensors to the model via metadata
        context.metadata.k_cache = k_cache
        context.metadata.v_cache = v_cache

        with torch.inference_mode():
            # Only pass VLM kwargs when the model supports them (images present)
            extra_kwargs = {}
            if context.images is not None:
                extra_kwargs["images"] = context.images
                extra_kwargs["image_grid_thw"] = context.image_grid_thw

            logits = self.model(
                input_ids=context.input_ids,
                attention_mask=context.attention_mask,
                position_ids=context.position_ids,
                metadata=context.metadata,
                **extra_kwargs,
            )

            # Extract last token logits for every query in the batch
            # Shape: (B, S_step, V) -> (B, V)
            # In Phase 3 DECODE, S_step is 1. In PREFILL, it is full prompt length.
            batch_size = context.input_ids.shape[0]
            last_logits = []
            for i in range(batch_size):
                # The index of the last token in the current STEP
                # (For decode, it is always index 0 as S_step=1)
                # But we use the context_lens to be general.
                step_len = context.input_ids.shape[1]  # S_step
                last_logits.append(logits[i, step_len - 1, :].float())

            batched_last_logits = torch.stack(last_logits)

            # Sample next tokens using full history for repetition penalty
            next_token_ids = self.sampler.select_batch(
                logits=batched_last_logits,
                all_generated_ids=context.token_histories,
                all_sampling_params=all_sampling_params,
            )

        return next_token_ids

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
        if hasattr(config, "vision_config"):
            return True
        architectures = getattr(config, "architectures", []) or []
        for arch in architectures:
            if "Vision" in arch or "VL" in arch or "ImageText" in arch:
                return True
        return False
