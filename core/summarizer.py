from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os 
import time

def get_llm():
    model = os.getenv("MISTRAL_MODEL", "open-mistral-nemo")
    return ChatMistralAI(model=model, mistral_api_key=os.getenv("MISTRAL_API_KEY"), temperature=0.3)


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)

def summarize(transcript: str) -> str:
    llm = get_llm()

    chunks = split_transcript(transcript)

    # If it's a short transcript (single chunk), summarize directly in one call
    if len(chunks) <= 1:
        direct_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert meeting summarizer. Provide a clear, professional meeting summary "
                "in structured bullet points highlighting the key discussions and conclusions.",
            ),
            ("human", "{text}"),
        ])
        direct_chain = direct_prompt | llm | StrOutputParser()
        return direct_chain.invoke({"text": transcript})

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely."),
        ("human", "{text}"),
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            time.sleep(1)  # Respect free tier rate limits
        chunk_summaries.append(map_chain.invoke({"text": chunk}))

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one final professional meeting summary in bullet points.",
        ),
        ("human", "{text}"),
    ])

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | combined_prompt | llm | StrOutputParser()
    )

    time.sleep(1)
    return combined_chain.invoke(combined)

def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | 
        ChatPromptTemplate.from_messages([
             (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title text, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )

    result = title_chain.invoke(transcript[:2000]).strip()
    return result.strip('"\'*#')




