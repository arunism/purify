from typing import Tuple, Optional
from abc import ABC, abstractmethod

import torch
import soundfile as sf
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class BaseLM(ABC):
    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is None:
            self.device: torch.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device
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
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                output_scores=True,
                return_dict_in_generate=True,
            )
        return self.tokenizer.decode(output.sequences[0]), torch.stack(
            output.scores, dim=1
        )


class BaseSpeechToText(ABC):
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device: torch.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device
        print(f"Using device: {self.device}")

        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(
            self.device
        )
        self.model.eval()  # Set model to evaluation mode

    @abstractmethod
    def generate_response(self, audio_path: str) -> str:
        pass

    @staticmethod
    def load_audio(audio_path: str) -> Tuple[torch.Tensor, int]:
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        return torch.from_numpy(audio), sample_rate

    @torch.no_grad()  # Disable gradient computation for inference
    def transcribe_audio(self, audio: torch.Tensor, sample_rate: int) -> str:
        input_values = self.processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        ).input_values
        input_values = input_values.to(self.device)

        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription
