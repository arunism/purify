import time
from typing import List, TypeVar

from models.base import BaseLM
from models.llama import BasicLlama2
from models.optim_llama import Llama2HallucinationReducer


BaseLMType = TypeVar("BaseLMType", BaseLM)


def compare_responses(
    query: str, basic_model: BaseLMType, advanced_model: BaseLMType
) -> None:
    print(f"Query: {query}\n")

    start_time = time.time()
    basic_response = basic_model.generate_response(query)
    basic_time = time.time() - start_time
    print(f"Basic Llama 2 Response (generated in {basic_time:.2f} seconds):")
    print(basic_response)
    print("\n" + "-" * 50 + "\n")

    start_time = time.time()
    advanced_response = advanced_model.generate_response(query)
    advanced_time = time.time() - start_time
    print(f"Hallucination-Reduced Response (generated in {advanced_time:.2f} seconds):")
    print(advanced_response)
    print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    basic_model: BaseLMType = BasicLlama2()
    advanced_model: BaseLMType = Llama2HallucinationReducer()

    # Set up document store and fact database
    advanced_model.set_document_store(
        [
            "Paris is the capital of France.",
            "The Eiffel Tower was completed in 1889.",
            "The Louvre Museum is located in Paris.",
            "France is a country in Western Europe.",
            "The Seine river runs through Paris.",
        ]
    )
    advanced_model.set_fact_database(
        [
            "Paris is the capital of France.",
            "The Eiffel Tower was constructed from 1887 to 1889.",
            "The Louvre Museum is the world's largest art museum.",
            "France shares borders with Belgium, Germany, Italy, and Spain.",
            "The Seine river is 777 kilometers long.",
        ]
    )

    queries: List[str] = [
        "What is the capital of France and when was the Eiffel Tower built?",
        "Tell me about the Louvre Museum and its location.",
        "Describe France's geography and major cities.",
    ]

    for query in queries:
        compare_responses(query, basic_model, advanced_model)
