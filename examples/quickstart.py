"""Quickstart — 5 lines to smarter search.

Run with: python examples/quickstart.py
"""

import refract

docs = [
    "Sort a Python list using the sorted() built-in function.",
    "Neural networks learn hierarchical representations of data.",
    "Retrieve relevant documents from a large corpus efficiently.",
    "Use cosine similarity to measure vector closeness in embedding space.",
    "Python decorators modify function behavior at definition time.",
    "Machine learning models require training data and compute.",
]

# One line to search — refract handles everything
results = refract.search("how do I sort things in Python", docs)

print("=" * 60)
print("refract quickstart")
print("=" * 60)
print()

for r in results:
    print(f"  {r.score:.3f}  {r.text}")
    # Show which metrics contributed
    breakdown = [(m.metric_name, round(m.weight, 2)) for m in r.provenance.metric_scores]
    print(f"         -> scored by: {breakdown}")
    print()

print("Query type detected:", results[0].provenance.query_type)
print("Space density:", results[0].provenance.space_density)
print("Router used:", results[0].provenance.router_name)
