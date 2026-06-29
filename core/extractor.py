# ActionableItems, decision, questions

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()


def get_llm():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set in environment / .env")
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.2,
    )


def build_chain(system_prompt: str):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{text}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an expert content analyst. Analyse the following transcript "
        "(which may be from a meeting, video, podcast, lecture, or discussion). "
        "Extract every action item, task, recommendation, or commitment mentioned. "
        "For each item provide:\n"
        "- What needs to be done\n"
        "- Who should do it (if mentioned, otherwise 'Not specified')\n"
        "- Deadline or timeframe (if mentioned, otherwise 'Not specified')\n\n"
        "Format as a numbered list. "
        "If the content contains no actionable items at all, summarise the top 3 "
        "key takeaways or recommendations from the content instead, labelled as "
        "'Key Takeaways'. Never respond with only 'No action items found'."
    )
    return chain.invoke({"text": transcript})


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert content analyst. Analyse the following transcript "
        "(which may be from a meeting, video, podcast, lecture, or discussion). "
        "Extract every key decision, conclusion, stance, or important point "
        "presented or agreed upon. Format as a numbered list. "
        "If no explicit decisions exist, list the 3–5 most important claims or "
        "conclusions from the content. Never respond with only 'No key decisions found'."
    )
    return chain.invoke({"text": transcript})


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert content analyst. Analyse the following transcript "
        "(which may be from a meeting, video, podcast, lecture, or discussion). "
        "Extract all open questions, unresolved issues, areas of uncertainty, "
        "or topics that need further investigation. Format as a numbered list. "
        "If no explicit questions exist, identify 3 interesting follow-up questions "
        "that a viewer or listener might want to explore further based on the content. "
        "Never respond with only 'No open questions found'."
    )
    return chain.invoke({"text": transcript})
