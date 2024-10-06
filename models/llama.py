from models.base import BaseLM


class BasicLlama2(BaseLM):
    def __init__(self, model_path: str = "meta-llama/Llama-2-7b-chat-hf"):
        super().__init__(model_path)

    def generate_response(self, query: str) -> str:
        return self._generate(query)
