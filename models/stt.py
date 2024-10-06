import torch

from models.base import BaseSpeechToText


class BasicSpeechToText(BaseSpeechToText):
    @torch.no_grad()
    def generate_response(self, audio_path: str) -> str:
        audio, sample_rate = self.load_audio(audio_path)
        transcription = self.transcribe_audio(audio, sample_rate)
        return transcription
