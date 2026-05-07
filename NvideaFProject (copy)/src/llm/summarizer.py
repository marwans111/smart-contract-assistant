"""
summarizer.py — LangChain map-reduce summarization for long contracts.
"""
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import List
from src.llm.qa_chain import _get_llm


MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=(
        "Summarize the following section of a legal contract in 3-5 bullet points:\n\n"
        "{text}\n\nSummary:"
    ),
)

COMBINE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=(
        "You are a legal document analyst. Below are summaries of individual sections "
        "of a contract. Write a concise overall summary (max 300 words) covering: "
        "parties involved, key obligations, payment terms, termination clauses, and risks.\n\n"
        "{text}\n\nFinal Summary:"
    ),
)


def summarize_document(chunks: List[Document]) -> str:
    """
    Summarize a list of document chunks using a map-reduce strategy.

    Args:
        chunks: List of Document chunks.

    Returns:
        A string containing the final summary.
    """
    llm = _get_llm()

    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=MAP_PROMPT,
        combine_prompt=COMBINE_PROMPT,
        verbose=False,
    )

    result = chain.invoke({"input_documents": chunks})
    return result["output_text"]
