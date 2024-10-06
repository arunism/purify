from typing import List, Tuple

import torch
from scipy import signal
import torch.nn.functional as F

from models.base import BaseSpeechToText


class GhostReductionSpeechToText(BaseSpeechToText):
    def __init__(self):
        super().__init__()
        self.confidence_threshold = 0.8
        self.common_replacements = {
            "their": ["there", "they're"],
            "your": ["you're", "yore"],
            "its": ["it's"],
            "to": ["too", "two"],
        }

    @torch.no_grad()
    def generate_response(self, audio_path: str) -> str:
        audio, sample_rate = self.load_audio(audio_path)
        audio = audio.to(self.device)

        filtered_audio = self.reduce_noise(audio, sample_rate)
        enhanced_audio = self.enhance_acoustic_model(filtered_audio)
        transcription, confidence_scores = self.transcribe_audio_with_confidence(
            enhanced_audio, sample_rate
        )
        corrected_transcription = self.apply_language_model(transcription)
        final_transcription = self.filter_low_confidence(
            corrected_transcription, confidence_scores
        )

        return final_transcription

    @staticmethod
    @torch.jit.script
    def reduce_noise(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        nyq = 0.5 * sample_rate
        low = 300 / nyq
        high = 3400 / nyq
        b, a = signal.butter(5, [low, high], btype="band")
        filtered_audio = signal.lfilter(b, a, audio.cpu().numpy())
        return torch.from_numpy(filtered_audio).to(audio.device)

    @staticmethod
    @torch.jit.script
    def enhance_acoustic_model(audio: torch.Tensor) -> torch.Tensor:
        audio = audio / torch.max(torch.abs(audio))
        pre_emphasis = 0.97
        emphasized_audio = F.pad(audio[1:] - pre_emphasis * audio[:-1], (1, 0))
        return emphasized_audio

    def transcribe_audio_with_confidence(
        self, audio: torch.Tensor, sample_rate: int
    ) -> Tuple[str, List[float]]:
        input_values = self.processor(
            audio.cpu().numpy(), sampling_rate=sample_rate, return_tensors="pt"
        ).input_values
        input_values = input_values.to(self.device)

        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        softmax_probs = F.softmax(logits, dim=-1)
        confidence_scores = (
            torch.max(softmax_probs, dim=-1).values.squeeze().cpu().tolist()
        )

        return transcription, confidence_scores

    def apply_language_model(self, transcription: str) -> str:
        words = transcription.split()
        for i, word in enumerate(words):
            for correct, alternatives in self.common_replacements.items():
                if word.lower() in alternatives:
                    words[i] = correct
        return " ".join(words)

    @staticmethod
    @torch.jit.script
    def filter_low_confidence(
        transcription: str, confidence_scores: List[float], threshold: float
    ) -> str:
        words = transcription.split()
        filtered_words = [
            word for word, score in zip(words, confidence_scores) if score >= threshold
        ]
        return " ".join(filtered_words)
