from __future__ import annotations

import time
from dataclasses import dataclass, field

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
from rank_bm25 import BM25Okapi


COLLECTION = "bonus_memory"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384


@dataclass
class UserFeatures:
    preferred_language: str = "vi"
    reading_speed_wpm: int = 210
    topic_affinity: str = "cloud"
    active_hours: str = "20:00-23:00"
    queries_last_hour: int = 0
    recent_topics: list[str] = field(default_factory=list)
    last_activity_minutes: int = 0

class HybridMemoryAgent:
    def __init__(self) -> None:
        self.embedder = TextEmbedding(model_name=EMBED_MODEL)
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        self.memories: list[dict] = []
        self.bm25 = BM25Okapi([["empty"]])
        self.features: dict[str, UserFeatures] = {"u_001": UserFeatures(
            preferred_language="vi/en mix",
            reading_speed_wpm=230,
            topic_affinity="cloud",
            active_hours="21:00-23:30",
            queries_last_hour=4,
            recent_topics=["kubernetes", "cloud", "security"],
            last_activity_minutes=8,
        )}

    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user."""
        doc_id = len(self.memories)
        topic = self._guess_topic(text)
        memory = {
            "doc_id": f"mem_{doc_id:04d}",
            "user_id": user_id,
            "text": text,
            "topic": topic,
            "created_at": time.time(),
        }
        self.memories.append(memory)
        vector = next(self.embedder.embed([text])).tolist()
        self.client.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(id=doc_id, vector=vector, payload=memory)],
        )
        self._rebuild_bm25()
        profile = self.features.setdefault(user_id, UserFeatures())
        profile.recent_topics = self._recent_topics(user_id)
        if profile.recent_topics:
            profile.topic_affinity = profile.recent_topics[0]

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve top-K memories + user profile features and return assembled context."""
        profile = self.features.setdefault(user_id, UserFeatures())
        profile.queries_last_hour += 1
        query_topic = self._guess_topic(query)
        if query_topic != "general":
            profile.recent_topics = [
                query_topic, *[t for t in profile.recent_topics if t != query_topic]
            ][:5]

        hits = self._hybrid_search(query, user_id=user_id, top_k=4)
        memory_lines = [
            f"- ({hit['topic']}, score={hit['score']:.4f}) {hit['text']}"
            for hit in hits
        ] or ["- No matching memory yet."]

        return "\n".join(
            [
                "ASSEMBLED CONTEXT",
                f"User: {user_id}",
                "Profile features:",
                f"- preferred_language: {profile.preferred_language}",
                f"- reading_speed_wpm: {profile.reading_speed_wpm}",
                f"- topic_affinity: {profile.topic_affinity}",
                f"- active_hours: {profile.active_hours}",
                "Recent activity features:",
                f"- queries_last_hour: {profile.queries_last_hour}",
                f"- recent_topics: {', '.join(profile.recent_topics) or 'none'}",
                f"- last_activity_minutes: {profile.last_activity_minutes}",
                "Retrieved episodic memories:",
                *memory_lines,
                "Suggested instruction to LLM:",
                "- Answer in Vietnamese unless the user asks for English; personalize using profile only when useful.",
            ]
        )
    def _hybrid_search(self, query: str, user_id: str, top_k: int = 4, rrf_k: int = 60) -> list[dict]:
        if not self.memories:
            return []
        semantic = self._semantic_search(query, user_id, depth=20)
        keyword = self._keyword_search(query, user_id, depth=20)
        scores: dict[str, float] = {}
        meta: dict[str, dict] = {}
        for ranked in (semantic, keyword):
            for rank, item in enumerate(ranked, start=1):
                scores[item["doc_id"]] = scores.get(item["doc_id"], 0.0) + 1.0 / (rrf_k + rank)
                meta[item["doc_id"]] = item
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [{**meta[doc_id], "score": score} for doc_id, score in ordered]

    def _semantic_search(self, query: str, user_id: str, depth: int) -> list[dict]:
        vector = next(self.embedder.embed([query])).tolist()
        result = self.client.query_points(
            collection_name=COLLECTION,
            query=vector,
            query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
            limit=depth,
        )
        return [dict(point.payload) for point in result.points]
    def _keyword_search(self, query: str, user_id: str, depth: int) -> list[dict]:
        scores = self.bm25.get_scores(self._tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        hits = []
        for i in ranked:
            memory = self.memories[i]
            if memory["user_id"] == user_id:
                hits.append(memory)
            if len(hits) >= depth:
                break
        return hits

    def _rebuild_bm25(self) -> None:
        tokens = [self._tokenize(memory["text"]) for memory in self.memories] or [["empty"]]
        self.bm25 = BM25Okapi(tokens)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().replace("/", " ").replace("-", " ").split()

    @staticmethod
    def _guess_topic(text: str) -> str:
        lowered = text.lower()
        topic_terms = {
            "kubernetes": ["kubernetes", "k8s", "pod", "cluster"],
            "cloud": ["cloud", "đám mây", "scaling", "scale", "autoscaling", "hạ tầng"],
            "security": ["security", "bảo mật", "zero trust", "iam", "secret"],
            "ai": ["ai", "llm", "embedding", "vector", "rag"],
        }
        for topic, terms in topic_terms.items():
            if any(term in lowered for term in terms):
                return topic
        return "general"

    def _recent_topics(self, user_id: str) -> list[str]:
        topics: list[str] = []
        for memory in reversed(self.memories):
            if memory["user_id"] == user_id and memory["topic"] not in topics:
                topics.append(memory["topic"])
            if len(topics) == 5:
                break
        return topics
