import streamlit as st
from duckduckgo_search import DDGS
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from dotenv import load_dotenv

import os
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LangChain ChatGroq model
llm = ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key=groq_api_key)

# Function to search DuckDuckGo for related news articles
def search_news(query, max_results=5):
    with DDGS() as ddgs:
        return [{"title": r["title"], "href": r["href"]} for r in ddgs.text(query, max_results=max_results)]

# Function to verify news using LangChain + Groq LLM
def verify_news(content, search_results):
    context = "\n".join([f"- {r['title']} ({r['href']})" for r in search_results])
    
    prompt = f"""
    You are an AI fact-checking assistant. Your task is to analyze the given news content and cross-check it with trusted news sources below.

    ### News Content:
    {content}
    
    ### Trusted Articles:
    {context}
    
    Based on the comparison, classify the news as:
    ✅ REAL: The claim is factually correct.
    ❌ FAKE: The claim is false or misleading.
    ⚠️ UNVERIFIED: Insufficient information to verify.

    Also, provide a confidence score (0-100%) and an explanation.
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return response.content

# Streamlit UI
st.title("📰 Fake News Detector")

news_input = st.text_area("Paste the news content or claim:")

if st.button("Authenticate"):
    if news_input.strip():
        st.write("🔍 Searching for related news...")
        search_results = search_news(news_input)
        
        if not search_results:
            st.error("❌ No relevant news found! Unable to verify.")
        else:
            st.write("✅ Relevant news articles found:")
            for idx, article in enumerate(search_results, 1):
                st.markdown(f"**{idx}. [{article['title']}]({article['href']})**")

            st.write("🤖 Analyzing with AI...")
            verification_result = verify_news(news_input, search_results)
            st.subheader("📢 Fake News Detection Result:")
            st.write(verification_result)
    else:
        st.warning("⚠️ Please enter news content to verify.")
