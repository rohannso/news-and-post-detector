# 📰 Fake News & AI Post Detector

This project is a **Streamlit-based web app** that helps detect **fake news** and **AI-generated images**. It leverages **OCR (Tesseract), AI analysis (Groq LLM), and fact-checking (DuckDuckGo)** to verify the authenticity of news content and social media posts.

---

## 🚀 Features

✅ **News Verification**:  
- Searches for related news using **DuckDuckGo**  
- Analyzes content with **AI (Groq LLM)**  
- Classifies news as **Real, Fake, or Unverified**  

✅ **Post Analysis**:  
- Extracts text from images using **OCR (Tesseract)**  
- Checks misinformation using **fact-checking sources**  
- Detects **AI-generated images**  

✅ **User-Friendly UI**  
- **Streamlit-based web app**  
- **Sidebar to switch between News & Post detection**  

---

## 📌 Installation

### **1️⃣ Clone the Repository**
```
git clone https://github.com/rohannso/news-and-post-detector.git
cd news-and-post-detector
pip install -r requirements.txt

```
Create a .env file in the project root and add:

GROQ_API_KEY=gsk_your_api_key_here

Running the App

streamlit run app.py