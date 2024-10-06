from typing import Optional

from models.base import BaseLM


class BasicLlama2(BaseLM):
    def __init__(
        self,
        model_path: str = "meta-llama/Llama-2-7b-chat-hf",
        device: Optional[str] = None,
    ):
        super().__init__(model_path, device)

    def generate_response(self, query: str) -> str:
        response, _ = self._generate(query)
        return response
