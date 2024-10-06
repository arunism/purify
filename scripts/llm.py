import os
import time
import argparse
from typing import List, TypeVar

import torch
from huggingface_hub import login

from models.base import BaseLM
from models.llama import BasicLlama2
from models.optim_llama import Llama2HallucinationReducer


BaseLMType = TypeVar("BaseLMType", bound=BaseLM)
login(token=os.environ.get("HF_TOKEN"))


models = {
    "basic": BasicLlama2,
    "advance": Llama2HallucinationReducer,
}


def main(query: str, model_name: str, device: str) -> None:
    device = torch.device(device)
    model = models.get(model_name)

    if not model:
        raise NotImplementedError(
            f"Model '{model}' NOT implemented! Available models are {list(models.keys())}"
        )

    model = model(device=device)

    if model_name == "advance":
        # Set up document store and fact database
        model.set_document_store(
            [
                "Paris is the capital of France.",
                "The Eiffel Tower was completed in 1889.",
                "The Louvre Museum is located in Paris.",
                "France is a country in Western Europe.",
                "The Seine river runs through Paris.",
            ]
        )
        model.set_fact_database(
            [
                "Paris is the capital of France.",
                "The Eiffel Tower was constructed from 1887 to 1889.",
                "The Louvre Museum is the world's largest art museum.",
                "France shares borders with Belgium, Germany, Italy, and Spain.",
                "The Seine river is 777 kilometers long.",
            ]
        )

    print(f"Query: {query}")
    print("-" * 50 + "\n")

    start_time = time.time()
    response = model.generate_response(query)
    exec_time = time.time() - start_time

    print(f"Execution Time: {exec_time:.2f} seconds)\n")
    print(response)
    print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--mode", type=str, default="advance")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    queries: List[str] = [
        "What is the capital of France and when was the Eiffel Tower built?",
        "Tell me about the Louvre Museum and its location.",
        "Describe France's geography and major cities.",
    ]

    for query in queries:
        main(query, args.mode, args.device)
