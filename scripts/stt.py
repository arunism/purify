import os
import time
import argparse
from typing import TypeVar

import torch
from huggingface_hub import login

from models.base import BaseSpeechToText
from models.stt import BasicSpeechToText
from models.optim_stt import GhostReductionSpeechToText


login(token=os.environ.get("HF_TOKEN"))
STTModelType = TypeVar("STTModelType", bound=BaseSpeechToText)


models = {
    "basic": BasicSpeechToText,
    "advance": GhostReductionSpeechToText,
}


def main(audio_path: str, model_name: str, device: str):
    device = torch.device(device)
    model = models.get(model_name)

    if not model:
        raise NotImplementedError(
            f"Model '{model}' NOT implemented! Available models are {list(models.keys())}"
        )

    model = model(device=device)

    start_time = time.time()
    result = model.generate_response(audio_path)
    exec_time = time.time() - start_time

    print(f"Execution Time: {exec_time:.2f} seconds)\n")
    print(f"STT Result: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--mode", type=str, default="advance")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(args.audio_path, args.mode, args.device)
