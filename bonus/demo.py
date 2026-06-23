from agent import HybridMemoryAgent


def seed(agent: HybridMemoryAgent) -> None:
    memories = [
        "I read a Vietnamese note about Kubernetes pods, deployments, and service discovery.",
        "Saved article: autoscaling cloud infrastructure when traffic spikes during a product launch.",
        "Notebook summary: vector search and BM25 can be fused with RRF for hybrid retrieval.",
        "Security note: cloud IAM, secret rotation, and zero trust reduce blast radius.",
        "Personal preference: I like concise Vietnamese summaries with a short English glossary.",
        "Reading log: Kubernetes HPA adjusts replicas from CPU and custom metrics.",
        "Document: cloud cost control needs budgets, tagging, and right-sizing.",
    ]
    for text in memories:
        agent.remember(text)


def main() -> None:
    agent = HybridMemoryAgent()
    seed(agent)

    queries = [
        "What have I read about Kubernetes?",
        "Recommend what to read next",
        "What am I focused on lately?",
        "Documents about scaling infrastructure?",
        "Give me a cloud security summary",
    ]

    for i, query in enumerate(queries, start=1):
        print("=" * 80)
        print(f"Query {i}: {query}")
        print(agent.recall(query))


if __name__ == "__main__":
    main()
