"""RAG pipeline example -- using refract as the retrieval step.

Shows how refract.search() can be used as a drop-in retrieval step
in a Retrieval-Augmented Generation pipeline. No external LLM
dependency required for this demo.

Run with: python examples/rag_pipeline.py
"""

import refract

# ── Knowledge base ──────────────────────────────────────────────────────────

knowledge_base = [
    "Our refund policy allows returns within 30 days of purchase. "
    "Items must be in original condition with tags attached.",
    "Shipping takes 3-5 business days for standard delivery. "
    "Express shipping is available for an additional $9.99.",
    "We accept Visa, Mastercard, American Express, and PayPal. "
    "All payments are processed securely through Stripe.",
    "Our customer support team is available Monday through Friday, "
    "9am to 6pm Eastern Time. Email support@example.com for help.",
    "Size exchanges are free for the first exchange. "
    "Please use the prepaid return label included in your package.",
    "International shipping is available to over 50 countries. "
    "Customs duties and taxes are the responsibility of the buyer.",
    "Gift cards can be purchased in denominations of $25, $50, and $100. "
    "They never expire and can be used for any product.",
    "We offer a price match guarantee within 14 days of purchase. "
    "Show us the lower price and we'll refund the difference.",
]


def retrieve(query: str, top_k: int = 3) -> list[str]:
    """Retrieve the most relevant documents for a query.

    This is the function you'd plug into your RAG pipeline,
    replacing vanilla cosine-similarity retrieval.

    Args:
        query: User's question.
        top_k: Number of documents to retrieve.

    Returns:
        List of relevant document texts.
    """
    results = refract.search(query, knowledge_base, top_k=top_k)
    return [r.text for r in results]


def format_context(docs: list[str]) -> str:
    """Format retrieved documents as context for an LLM."""
    return "\n\n".join(f"[{i + 1}] {doc}" for i, doc in enumerate(docs))


# ── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    queries = [
        "Can I return something I bought last week?",
        "How long does shipping take?",
        "Do you ship internationally?",
    ]

    print("=" * 60)
    print("refract as a RAG retrieval step")
    print("=" * 60)

    for query in queries:
        print(f"\n[Q] Query: {query}")
        print("-" * 50)

        context_docs = retrieve(query, top_k=2)
        context = format_context(context_docs)

        print(f"[CTX] Retrieved context:\n{context}")

        # In a real pipeline, you'd now feed this to an LLM:
        # prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        # answer = llm.generate(prompt)
        print()
