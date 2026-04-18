from typing import Any, Dict, List, Optional

from nano_inference.input_processor.base import (
    BaseInputProcessor,
    GenerationInputs,
    register_input_processor,
)


@register_input_processor
class BareInputProcessor(BaseInputProcessor):
    name = "bare"

    def encode(
        self,
        messages: List[Dict[str, str]],
        max_prompt_tokens: Optional[int] = None,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> GenerationInputs:
        max_tokens = (
            max_prompt_tokens if max_prompt_tokens is not None else self.max_seq_len
        )

        text_content = ""
        for msg in messages:
            text_content += msg["content"]

        prompt_token_ids = self.tokenizer.encode(
            text_content,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )

        if len(prompt_token_ids) > max_tokens:
            prompt_token_ids = prompt_token_ids[-max_tokens:]

        return GenerationInputs(prompt_token_ids=prompt_token_ids)
