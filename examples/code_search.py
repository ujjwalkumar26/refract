"""Code search example -- refract detects code queries automatically.

Demonstrates that refract detects "code" query type and adjusts
metric weights accordingly (boosting BM25 for keyword matching).

Run with: python examples/code_search.py
"""

import refract

# ── Code docstring corpus ───────────────────────────────────────────────────

code_docs = [
    "def sorted(iterable, key=None, reverse=False): Return a new sorted list from the items in iterable.",
    "def filter(function, iterable): Construct an iterator from elements of iterable for which function returns true.",
    "def map(function, iterable): Apply function to every item of iterable and return a list of results.",
    "def reduce(function, iterable, initializer=None): Apply function of two arguments cumulatively to the items of iterable.",
    "def enumerate(iterable, start=0): Return an enumerate object yielding pairs of (index, element).",
    "def zip(*iterables): Make an iterator that aggregates elements from each of the iterables.",
    "def reversed(seq): Return a reverse iterator over the values of the given sequence.",
    "def len(obj): Return the number of items in a container.",
    "def range(start, stop, step=1): Generate a sequence of numbers from start to stop with a given step.",
    "def isinstance(obj, classinfo): Return True if the object is an instance of the classinfo argument.",
    "class list: Built-in mutable sequence type. Supports append, extend, sort, reverse, and other operations.",
    "class dict: Built-in mapping type. Stores key-value pairs with O(1) average lookup time.",
    "class set: Built-in unordered collection of unique elements. Supports union, intersection, difference.",
]


# ── Search with different query types ───────────────────────────────────────

if __name__ == "__main__":
    queries = [
        # This should be detected as "code"
        "def sort_list(arr) -> list:",
        # This should be "keyword"
        "sort list python",
        # This should be "natural_language"
        "How can I iterate through a list and transform each element?",
    ]

    print("=" * 60)
    print("refract code search -- query type detection")
    print("=" * 60)

    for query in queries:
        results = refract.search(query, code_docs, top_k=3)
        prov = results[0].provenance

        print(f"\n[SEARCH] Query: {query!r}")
        print(f"   Detected type: {prov.query_type}")
        print(f"   Metric weights:")
        for ms in prov.metric_scores:
            bar = "#" * int(ms.weight * 30)
            print(f"     {ms.metric_name:15s} w={ms.weight:.2f} {bar}")
        print(f"   Top result: {results[0].text[:70]}...")
        print(f"   Score: {results[0].score:.3f}")
