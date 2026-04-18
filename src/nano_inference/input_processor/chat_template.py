from typing import Any, Dict, List, Optional

from nano_inference.input_processor.base import (
    BaseInputProcessor,
    GenerationInputs,
    register_input_processor,
)


@register_input_processor
class ChatTemplateInputProcessor(BaseInputProcessor):
    name = "chat_template"

    def encode(
        self,
        messages: List[Dict[str, str]],
        max_prompt_tokens: Optional[int] = None,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> GenerationInputs:
        max_tokens = (
            max_prompt_tokens if max_prompt_tokens is not None else self.max_seq_len
        )

        prompt_token_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            **kwargs,
        )

        # Handle case where tokenizer returns a BatchEncoding object
        if hasattr(prompt_token_ids, "input_ids"):
            prompt_token_ids = prompt_token_ids.input_ids
        # Handle case where tokenizer returns a dict
        elif isinstance(prompt_token_ids, dict) and "input_ids" in prompt_token_ids:
            prompt_token_ids = prompt_token_ids["input_ids"]

        if len(prompt_token_ids) > max_tokens:
            prompt_token_ids = prompt_token_ids[-max_tokens:]

        return GenerationInputs(prompt_token_ids=prompt_token_ids)
