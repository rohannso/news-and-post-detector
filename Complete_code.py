import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image
import time
from duckduckgo_search import DDGS
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import easyocr

# Get API key from Streamlit secrets
groq_api_key = st.secrets["groq"]["api_key"]

# ✅ Set Up Groq LLM
llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")

# ✅ Extract Text from Image (OCR)
def extract_text(image_path):
    reader = easyocr.Reader(['en'])  # Set language to English
    result = reader.readtext(image_path, detail=0)  # Extract text
    return " ".join(result)

# ✅ Fact-Check Using DuckDuckGo Search
def search_fact_check(query):
    search_results = []
    with DDGS() as ddgs:
        results = ddgs.text(f"{query} site:snopes.com OR site:politifact.com OR site:reuters.com", max_results=5)
        for r in results:
            search_results.append(f"[{r['title']}]({r['href']})")
    return "\n".join(search_results) if search_results else "No fact-checking results found."

# ✅ AI Misinformation Analysis
def analyze_misinformation(text):
    prompt = f"""
    Analyze the following text and determine if it contains misinformation:
    {text}
    """
    response = llm.predict(prompt)
    return response

# ✅ AI-Generated Image Detection
def detect_ai_generated(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return "AI-Generated Image Detected" if laplacian < 50 else "Likely a Real Image"

# ✅ Summarize Findings
def summarize_findings(text_analysis, fact_check_results, image_analysis):
    prompt = f"""
    Given the following analysis, summarize if the content is real or fake:
    - Text Analysis: {text_analysis}
    - Fact Check Results: {fact_check_results}
    - Image Analysis: {image_analysis}
    Provide a clear and concise decision.
    """
    response = llm.predict(prompt)
    return response

# ✅ Search News Articles
def search_news(query, max_results=5):
    with DDGS() as ddgs:
        return [{"title": r["title"], "href": r["href"]} for r in ddgs.text(query, max_results=max_results)]

# ✅ Verify News Using AI
def verify_news(content, search_results):
    context = "\n".join([f"- {r['title']} ({r['href']})" for r in search_results])
    prompt = f"""
    You are an AI fact-checking assistant. Analyze the given news content and compare it with trusted sources below.
    
    ### News Content:
    {content}
    
    ### Trusted Articles:
    {context}
    
    Classify the news as:
    ✅ REAL
    ❌ FAKE
    ⚠️ UNVERIFIED
    Provide a confidence score and an explanation.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# ✅ Streamlit UI with Sidebar
st.title("🕵️ Fake News & AI Image Detector")

option = st.sidebar.radio("Select Detection Mode", ["Post Detection", "News Detection"])

if option == "Post Detection":
    st.header("📸 Post Detection")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extracting text with a spinner
        with st.spinner("🔍 Extracting text from image..."):
            time.sleep(2)  # Simulating delay
            extracted_text = extract_text("uploaded_image.jpg")
        st.write(extracted_text)

        # Fact-checking with a spinner
        with st.spinner("🌐 Fact-checking with DuckDuckGo..."):
            time.sleep(3)  # Simulating delay
            fact_results = search_fact_check(extracted_text)
        st.write(fact_results)

        # AI Misinformation Analysis with a spinner
        with st.spinner("🧠 Analyzing misinformation..."):
            time.sleep(2)  # Simulating delay
            ai_analysis = analyze_misinformation(extracted_text)
        st.write(ai_analysis)

        # AI-Generated Image Detection with a spinner
        with st.spinner("🎭 Checking for AI-generated images..."):
            time.sleep(2)  # Simulating delay
            image_result = detect_ai_generated("uploaded_image.jpg")
        st.write(image_result)

        # Summarizing the findings with a spinner
        with st.spinner("📊 Summarizing findings..."):
            time.sleep(2)  # Simulating delay
            final_decision = summarize_findings(ai_analysis, fact_results, image_result)
        st.subheader("📢 Final Decision:")
        st.write(final_decision)

elif option == "News Detection":
    st.header("📰 News Detection")
    news_input = st.text_area("Paste the news content or claim:")
    if st.button("Authenticate"):
        if news_input.strip():
            with st.spinner("🔍 Searching for related news..."):
                time.sleep(3)  # Simulating delay
                search_results = search_news(news_input)

            if not search_results:
                st.error("❌ No relevant news found! Unable to verify.")
            else:
                st.write("✅ Relevant news articles found:")
                for idx, article in enumerate(search_results, 1):
                    st.markdown(f"**{idx}. [{article['title']}]({article['href']})**")

                with st.spinner("🤖 Analyzing with AI..."):
                    time.sleep(2)  # Simulating delay
                    verification_result = verify_news(news_input, search_results)

                st.subheader("📢 Fake News Detection Result:")
                st.write(verification_result)
        else:
            st.warning("⚠️ Please enter news content to verify.")
