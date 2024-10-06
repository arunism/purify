from typing import List, Tuple, Dict, Optional

import torch
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from models.base import BaseLM


class Llama2HallucinationReducer(BaseLM):
    def __init__(
        self,
        model_path: str = "meta-llama/Llama-2-7b-chat-hf",
        device: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        super().__init__(model_path, device)
        self.embedder: SentenceTransformer = SentenceTransformer(embedding_model).to(
            self.device
        )
        self.document_store: List[str] = []
        self.fact_database: List[str] = []
        self.doc_embeddings: Optional[torch.Tensor] = None
        self.fact_embeddings: Optional[torch.Tensor] = None
        self.knowledge_graph: nx.Graph = nx.Graph()

    def set_document_store(self, documents: List[str]) -> None:
        self.document_store = documents
        self.doc_embeddings = self.embedder.encode(documents, convert_to_tensor=True)

    def set_fact_database(self, facts: List[str]) -> None:
        self.fact_database = facts
        self.fact_embeddings = self.embedder.encode(facts, convert_to_tensor=True)

    def build_knowledge_graph(self, triples: List[Tuple[str, str, str]]) -> None:
        for subject, predicate, obj in triples:
            self.knowledge_graph.add_edge(subject, obj, relation=predicate)

    def knowledge_graph_generation(self, query: str) -> str:
        entities = self._extract_entities(query)
        relevant_nodes = set()
        for entity in entities:
            if entity in self.knowledge_graph:
                relevant_nodes.update(nx.neighbors(self.knowledge_graph, entity))

        subgraph = self.knowledge_graph.subgraph(relevant_nodes)
        graph_info = "\n".join(
            [
                f"{u} - {self.knowledge_graph[u][v]['relation']} -> {v}"
                for u, v in subgraph.edges()
            ]
        )
        augmented_prompt = (
            f"Using the following knowledge:\n{graph_info}\n\nAnswer: {query}"
        )

        return self._generate(augmented_prompt)

    def retrieval_augmented_generation(self, query: str) -> str:
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

    def fact_verification(self, generated_text: str) -> str:
        facts = generated_text.split(".")
        verified_text = generated_text

        fact_embeddings = self.embedder.encode(facts, convert_to_tensor=True)
        similarities = cosine_similarity(
            fact_embeddings.cpu().numpy(), self.fact_embeddings.cpu().numpy()
        )

        for i, fact in enumerate(facts):
            if np.max(similarities[i]) < 0.7:
                verified_text = verified_text.replace(fact, f"[UNVERIFIED: {fact}]")

        return verified_text

    def prompt_engineering(
        self, query: str, examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        engineered_prompt = (
            "Please provide an accurate and factual response to the following question. "
            "If you're unsure about any part of the answer, please indicate your uncertainty. "
            "Avoid speculation and stick to verified information.\n\n"
        )

        # Add few-shot examples
        if examples:
            for example in self.few_shot_examples:
                engineered_prompt += (
                    f"Question: {example['query']}\nAnswer: {example['response']}\n\n"
                )

        # Add chain-of-thought prompting
        engineered_prompt += (
            f"Now, please answer the following question step by step:\n{query}\n"
            "1. Identify the key elements of the question.\n"
            "2. Recall relevant facts and information.\n"
            "3. Reason through the answer logically.\n"
            "4. Provide a concise and accurate response.\n\n"
            "Answer: "
        )

        return engineered_prompt

    def output_calibration(
        self, response: str, scores: torch.Tensor, confidence_threshold: float = 0.9
    ) -> str:
        token_probabilities = scores.softmax(dim=-1).max(dim=-1).values
        if torch.any(token_probabilities < confidence_threshold):
            uncertain_response = (
                "I'm not entirely certain, but based on the information available: "
                + response
            )
            return uncertain_response
        return response

    def generate_response(
        self, query: str, examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        engineered_prompt = self.prompt_engineering(query, examples)
        rag_response, rag_scores = self.retrieval_augmented_generation(
            engineered_prompt
        )
        kg_response, kg_scores = self.knowledge_graph_generation(engineered_prompt)
        calibrated_rag_response = self.output_calibration(rag_response, rag_scores)
        calibrated_kg_response = self.output_calibration(kg_response, kg_scores)
        combined_response = f"{calibrated_rag_response}\n\nAdditional information from knowledge graph:\n{calibrated_kg_response}"
        verified_response = self.fact_verification(combined_response)
        return verified_response

    def _extract_entities(self, text: str) -> List[str]:
        # This is a simplified entity extraction. In a real-world scenario,
        # we would use a more sophisticated NER (Named Entity Recognition) system.
        return [word for word in text.split() if word[0].isupper()]
