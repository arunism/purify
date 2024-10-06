from abc import ABC, abstractmethod

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM


class BaseLM(ABC):
    def __init__(self, model_path: str):
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(model_path).to(
            self.device
        )
        self.tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(model_path)

    @abstractmethod
    def generate_response(self, query: str) -> str:
        pass

    def _generate(self, prompt: str, max_length: int = 200) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(output[0])
