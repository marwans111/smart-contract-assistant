"""
evaluation/run_eval.py
Evaluate the RAG pipeline using RAGAS metrics:
  - Faithfulness
  - Answer Relevancy
  - Context Recall
"""
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

# ── Sample evaluation data ────────────────────────────────────────────────────
# Replace with real contract Q&A pairs for proper evaluation

EVAL_DATA = {
    "question": [
        "What is the payment due date?",
        "Who are the parties in this agreement?",
        "What are the termination conditions?",
    ],
    "answer": [
        "Payment is due within 30 days of invoice.",
        "The agreement is between Acme Corp and Beta LLC.",
        "Either party may terminate with 60 days written notice.",
    ],
    "contexts": [
        ["Payment shall be made within 30 days of the invoice date."],
        ["This Agreement is entered into by Acme Corp ('Client') and Beta LLC ('Vendor')."],
        ["This Agreement may be terminated by either party upon 60 days written notice."],
    ],
    "ground_truth": [
        "Payment is due within 30 days of invoice.",
        "Acme Corp and Beta LLC.",
        "60 days written notice.",
    ],
}


def run_evaluation():
    dataset = Dataset.from_dict(EVAL_DATA)

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
    )

    print("\n===== RAGAS Evaluation Results =====")
    print(results)
    print("=====================================\n")
    return results


if __name__ == "__main__":
    run_evaluation()
