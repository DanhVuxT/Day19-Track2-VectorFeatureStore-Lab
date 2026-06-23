import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.main import app
from app.search import SearchHit
from app.feast_repo.feature_views import (
    item_popularity_features,
    query_velocity_features,
    user_profile_features,
)
from scripts.benchmark import precision_at_k


def test_fastapi_app_exposes_search_endpoint() -> None:
    paths = {route.path for route in app.routes}
    assert "/search" in paths
    assert "/healthz" in paths


def test_search_hit_serializes_response_shape() -> None:
    hit = SearchHit(doc_id="cloud_001", title="Cloud", text="Autoscaling", score=0.42)
    assert hit.dict() == {
        "doc_id": "cloud_001",
        "title": "Cloud",
        "text": "Autoscaling",
        "score": 0.42,
    }


def test_precision_at_k_counts_relevant_hits() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c", "x"}
    assert precision_at_k(retrieved, relevant, k=4) == 0.5


def test_feast_repo_defines_three_online_feature_views() -> None:
    views = [user_profile_features, item_popularity_features, query_velocity_features]
    assert {view.name for view in views} == {
        "user_profile_features",
        "item_popularity_features",
        "query_velocity_features",
    }
    assert all(view.online for view in views)
