from dataclasses import dataclass


@dataclass
class SamplingParams:
    max_new_tokens: int = 256
    temperature: float = 0.7  # 0 = greedy, higher = more random
    top_k: int = -1  # only consider top_k tokens, -1 = disable and consider all tokens
    top_p: float = (
        1.0  # Nucleus sampling, only consider tokens with cumulative prob <= top_p
    )
    repetition_penalty: float = 1.0  # penalize repeated tokens, 1.0 = no penalty
