import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_mistralai import ChatMistralAI

from core.vector_store import build_vector_store, get_retriever, load_vector_store

load_dotenv()

RAG_SYSTEM_PROMPT = """You are an expert meeting assistant. Answer the user's question \
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: \
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}"""


def get_llm():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set in environment / .env")
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.3,
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def _build_rag_chain_from_retriever(retriever):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    @chain
    def retrieve_and_format(question: str) -> dict:
        docs = retriever.invoke(question)
        return {"context": format_docs(docs), "question": question}

    return retrieve_and_format | prompt | llm | StrOutputParser()


def build_rag_chain(transcript: str, progress_fn=None):
    vector_store = build_vector_store(transcript, progress_fn=progress_fn)
    retriever = get_retriever(vector_store, k=4)
    return _build_rag_chain_from_retriever(retriever)


def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=4)
    return _build_rag_chain_from_retriever(retriever)


def ask_question(rag_chain, question: str) -> str:
    print(f"Question : {question}")
    answer = rag_chain.invoke(question)
    print(f"answer : {answer}")
    return answer
