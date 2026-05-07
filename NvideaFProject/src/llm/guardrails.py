"""
guardrails.py — Input validation and output grounding checks.
"""

BLOCKED_TOPICS = [
    "weapon", "bomb", "illegal", "hack", "password", "personal data",
    "credit card", "social security"
]

DISCLAIMER = (
    "\n\n⚠️ *This answer is based solely on the uploaded document and is "
    "for informational purposes only. It does not constitute legal advice.*"
)


def check_input(query: str) -> str | None:
    """
    Validate user input. Returns an error message if the query is blocked,
    or None if the query is acceptable.
    """
    query_lower = query.lower()
    for topic in BLOCKED_TOPICS:
        if topic in query_lower:
            return (
                f"⛔ Sorry, I can't help with queries related to '{topic}'. "
                "Please ask questions directly related to your uploaded contract."
            )
    if len(query.strip()) < 5:
        return "⚠️ Please enter a more specific question."
    return None


def apply_output_guardrail(answer: str, source_docs: list) -> str:
    """
    Append disclaimer and source page numbers to the final answer.
    """
    if not source_docs:
        return answer + "\n\n⚠️ *No source chunks were retrieved. Answer may be unreliable.*"

    pages = sorted({
        doc.metadata.get("page", "?") for doc in source_docs
    })
    source_note = f"\n\n📎 *Sources: page(s) {', '.join(str(p) for p in pages)}*"

    return answer + source_note + DISCLAIMER
