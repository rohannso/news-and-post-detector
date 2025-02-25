import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image
import requests
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS
import time

# ✅ Configure Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✅ Set Up Groq LLM
groq_api_key = "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"
llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")

# ✅ 1. Extract Text from Image (OCR)
def extract_text(image_path):
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text.strip()

# ✅ 2. Fact-Check Using DuckDuckGo Search
def search_fact_check(query):
    search_results = []
    with DDGS() as ddgs:
        time.sleep(3)
        results = ddgs.text(f"{query} site:snopes.com OR site:politifact.com OR site:reuters.com", max_results=5)
        for r in results:
            search_results.append(f"[{r['title']}]({r['href']})")
    
    return "\n".join(search_results) if search_results else "No fact-checking results found."

# ✅ 3. Misinformation Analysis with AI
def analyze_misinformation(text):
    prompt = f"Analyze the following text and determine if it contains misinformation:\n\n{text}"
    response = llm.predict(prompt)
    return response

# ✅ 4. AI-Generated Image Detection
def detect_ai_generated(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return "AI-Generated Image Detected" if laplacian < 50 else "Likely a Real Image"

# ✅ 5. Summarize Findings for Decision-Making
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

# ✅ 6. Build Streamlit UI
st.title("🕵️ Fake News & AI Image Detector")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Save uploaded image
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract Text
    st.subheader("🔍 Extracting Text from Image...")
    extracted_text = extract_text("uploaded_image.jpg")
    st.write(extracted_text)

    # Fact-Checking
    st.subheader("🌐 Fact-Checking with DuckDuckGo...")
    fact_results = search_fact_check(extracted_text)
    st.write(fact_results)

    # AI Analysis
    st.subheader("🧠 AI Misinformation Analysis:")
    ai_analysis = analyze_misinformation(extracted_text)
    st.write(ai_analysis)

    # AI-Generated Image Detection
    st.subheader("🎭 AI-Generated Image Detection:")
    image_result = detect_ai_generated("uploaded_image.jpg")
    st.write(image_result)

    # Final Decision
    st.subheader("📊 Final Decision:")
    final_decision = summarize_findings(ai_analysis, fact_results, image_result)
    st.write(final_decision)
