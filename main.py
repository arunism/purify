import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from huggingface_hub import login


login(token="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")


class BasicLlama2:
    def __init__(self, model_path="meta-llama/Llama-2-7b-chat-hf"):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model = LlamaForCausalLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)

    def generate_response(self, query):
        input_ids = self.tokenizer.encode(query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=200)
        return self.tokenizer.decode(output[0])


class SimplifiedLlama2HallucinationReducer:
    def __init__(
        self,
        model_path="meta-llama/Llama-2-7b-chat-hf",
        embedding_model="all-MiniLM-L6-v2",
    ):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model = LlamaForCausalLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.embedder = SentenceTransformer(embedding_model).to(self.device)
        self.document_store = []
        self.fact_database = []
        self.doc_embeddings = None
        self.fact_embeddings = None

    def set_document_store(self, documents):
        self.document_store = documents
        self.doc_embeddings = self.embedder.encode(documents, convert_to_tensor=True)

    def set_fact_database(self, facts):
        self.fact_database = facts
        self.fact_embeddings = self.embedder.encode(facts, convert_to_tensor=True)

    def retrieval_augmented_generation(self, query):
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        similarities = cosine_similarity(
            query_embedding.cpu().numpy().reshape(1, -1),
            self.doc_embeddings.cpu().numpy(),
        )[0]

        k = 3
        top_k_indices = similarities.argsort()[-k:][::-1]
        retrieved_docs = [self.document_store[i] for i in top_k_indices]

        augmented_prompt = "Based on the following information:\n"
        augmented_prompt += "\n".join(retrieved_docs)
        augmented_prompt += f"\n\nAnswer the question: {query}"

        return self._generate(augmented_prompt)

    def fact_verification(self, generated_text):
        facts = generated_text.split(".")
        verified_text = generated_text

        fact_embeddings = self.embedder.encode(facts, convert_to_tensor=True)
        similarities = cosine_similarity(
            fact_embeddings.cpu().numpy(), self.fact_embeddings.cpu().numpy()
        )

        for i, fact in enumerate(facts):
            if np.max(similarities[i]) < 0.8:
                verified_text = verified_text.replace(fact, f"[UNVERIFIED: {fact}]")

        return verified_text

    def generate_reliable_response(self, query):
        grounded_response = self.retrieval_augmented_generation(query)
        verified_response = self.fact_verification(grounded_response)
        return verified_response

    def _generate(self, prompt, max_length=200):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(output[0])


def compare_responses(query, basic_model, advanced_model):
    print(f"Query: {query}\n")

    start_time = time.time()
    basic_response = basic_model.generate_response(query)
    basic_time = time.time() - start_time
    print(f"Basic Llama 2 Response (generated in {basic_time:.2f} seconds):")
    print(basic_response)
    print("\n" + "-" * 50 + "\n")

    start_time = time.time()
    advanced_response = advanced_model.generate_reliable_response(query)
    advanced_time = time.time() - start_time
    print(f"Hallucination-Reduced Response (generated in {advanced_time:.2f} seconds):")
    print(advanced_response)
    print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    basic_model = BasicLlama2()
    advanced_model = SimplifiedLlama2HallucinationReducer()

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

    queries = [
        "What is the capital of France and when was the Eiffel Tower built?",
        "Tell me about the Louvre Museum and its location.",
        "Describe France's geography and major cities.",
    ]

    for query in queries:
        compare_responses(query, basic_model, advanced_model)
