import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image
from duckduckgo_search import DDGS
import os
import groq
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from io import BytesIO

# ✅ Configure Tesseract (Make sure it's installed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✅ Groq API Key (Use environment variable for security)
groq_api_key = os.getenv("GROQ_API_KEY")  
llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")

# ✅ OCR Extraction Agent
def extract_text_from_image(image_path):
    if os.path.exists(image_path):
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image)
        return extracted_text.strip()
    return "Image file not found."

ocr_tool = Tool(
    name="OCR Extraction Agent",
    func=extract_text_from_image,
    description="Extracts text from images using OCR."
)

# ✅ Fact-Checking Agent
def search_fact_check_sites(query):
    with DDGS() as ddgs:
        search_results = ddgs.text(query + " site:snopes.com OR site:politifact.com OR site:bbc.com OR site:reuters.com")
    return search_results

fact_check_tool = Tool(
    name="Fact-Check Agent",
    func=search_fact_check_sites,
    description="Searches for fact-checked articles related to the query."
)

# ✅ AI Misinformation Detection Agent
def analyze_text_misinformation(text):
    prompt = f"Analyze the following text for misinformation and tell me if it's likely fake:\n\n{text}"
    response = llm.predict(prompt)
    return response

misinfo_tool = Tool(
    name="Misinformation AI Agent",
    func=analyze_text_misinformation,
    description="Analyzes text to detect misinformation."
)

# ✅ AI-Generated Image Detection
def detect_ai_generated_image(image_path):
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        return "AI-Generated Image Detected" if laplacian < 50 else "Likely a Real Image"
    return "Image file not found."

image_analysis_tool = Tool(
    name="Image Forensics Agent",
    func=detect_ai_generated_image,
    description="Detects if an image is AI-generated."
)

# ✅ Create LangChain Agent
tools = [ocr_tool, fact_check_tool, misinfo_tool, image_analysis_tool]
memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# ✅ Streamlit UI
st.title("🕵️ Fake Image & Post Detector with AI Agents")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded file correctly
    image_path = "uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ✅ Step 1: Extract Text from Image
    st.subheader("🔍 Extracting Text from Image...")
    extracted_text = agent.run(f"Extract text from {image_path}")
    st.write(extracted_text)

    # ✅ Step 2: Fact-Checking
    st.subheader("🌐 Fact-Checking...")
    search_results = agent.run(f"Check if this is misinformation: {extracted_text}")
    st.write(search_results)

    # ✅ Step 3: AI Misinformation Analysis
    st.subheader("🧠 AI Misinformation Analysis:")
    ai_analysis = agent.run(f"Analyze this extracted text: {extracted_text}")
    st.write(ai_analysis)

    # ✅ Step 4: AI-Generated Image Detection
    st.subheader("🎭 AI-Generated Image Detection:")
    image_result = agent.run(f"Analyze {image_path} for AI manipulation")
    st.write(image_result)

    # ✅ Step 5: Final Decision
    st.subheader("📊 Final Decision:")
    fake_score = 50 if "misinformation" in ai_analysis.lower() else 0
    fake_score += 50 if "AI-Generated Image Detected" in image_result else 0
    st.progress(fake_score / 100)

    if fake_score >= 50:
        st.error("⚠️ This post/image might contain misinformation!")
    else:
        st.success("✅ This post/image appears to be legitimate.")
