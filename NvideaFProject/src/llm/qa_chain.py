"""
qa_chain.py — LangChain conversational RAG chain with source citations.
"""
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from src.utils.config import LLM_PROVIDER, OPENAI_API_KEY


QA_PROMPT_TEMPLATE = """You are a helpful legal document assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say: "I couldn't find that in the document."
Do not make up information.

Context:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=QA_PROMPT_TEMPLATE,
)


def _get_llm():
    """Return LLM based on config."""
    if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    else:
        from langchain_huggingface import HuggingFacePipeline
        from transformers import pipeline
        pipe = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_new_tokens=512,
        )
        return HuggingFacePipeline(pipeline=pipe)


def build_qa_chain(retriever):
    """
    Build a ConversationalRetrievalChain with memory.

    Args:
        retriever: A LangChain retriever.

    Returns:
        A LangChain chain ready to invoke.
    """
    llm = _get_llm()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )

    print("[QA Chain] Built successfully")
    return chain


def ask(chain, question: str) -> dict:
    """
    Run a question through the QA chain.

    Args:
        chain: The ConversationalRetrievalChain.
        question: User's question string.

    Returns:
        Dict with 'answer' and 'source_documents'.
    """
    result = chain.invoke({"question": question})
    return result
