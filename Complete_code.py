import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from duckduckgo_search import DDGS
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import easyocr
from deep_translator import GoogleTranslator
from langdetect import detect

# Set page configuration
st.set_page_config(
    page_title="Fake News & AI Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        border: 2px solid #FF4B4B;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    h1 {
        color: #1E3D59;
        text-align: center;
        padding-bottom: 1rem;
        border-bottom: 2px solid #FF4B4B;
    }
    h2 {
        color: #1E3D59;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Get API key from Streamlit secrets
groq_api_key = st.secrets["groq"]["api_key"]


# ✅ Set Up Groq LLM
llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")

# ✅ Detect Language from Text
def detect_language(text):
    try:
        lang_code = detect(text)
        language_names = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 
            'de': 'German', 'zh-cn': 'Chinese', 'ar': 'Arabic',
            'hi': 'Hindi', 'ru': 'Russian', 'pt': 'Portuguese',
            'it': 'Italian', 'ja': 'Japanese', 'ko': 'Korean'
        }
        return lang_code, language_names.get(lang_code, 'Unknown')
    except:
        return 'en', 'English (default)'

# ✅ Translation Function
def translate_text(text, source_language, target_language):
    if source_language == target_language:
        return text
    try:
        translated = GoogleTranslator(source=source_language, target=target_language).translate(text)
        return translated
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

# ✅ Extract Text from Image (OCR) with Multiple Language Support
def extract_text(image_path, languages=['en']):
    reader = easyocr.Reader(languages)  # Set languages dynamically
    result = reader.readtext(image_path, detail=0)  # Extract text
    return " ".join(result)

# ✅ Fact-Check Using DuckDuckGo Search with Regional Sources
def search_fact_check(query, language='en'):
    search_results = []
    region_map = {'en': 'us', 'es': 'es', 'fr': 'fr', 'de': 'de', 'ar': 'ae', 
                 'zh-cn': 'cn', 'hi': 'in', 'ru': 'ru', 'pt': 'br', 'it': 'it',
                 'ja': 'jp', 'ko': 'kr'}
    region = region_map.get(language, 'us')
    
    sites = {
        'en': "site:snopes.com OR site:politifact.com OR site:reuters.com OR site:factcheck.org",
        'es': "site:maldita.es OR site:newtral.es OR site:efe.com OR site:verificat.cat",
        'fr': "site:liberation.fr OR site:lemonde.fr OR site:franceinfo.fr OR site:20minutes.fr",
        'de': "site:correctiv.org OR site:faktenfinder.tagesschau.de OR site:dpa-factchecking.com",
        'hi': "site:altnews.in OR site:factchecker.in OR site:boomlive.in",
        'pt': "site:aosfatos.org OR site:lupa.uol.com.br OR site:estadao.com.br",
        'ru': "site:fakecheck.stopfake.org OR site:factcheck.kz",
        'zh-cn': "site:fact-checking-china.com OR site:factcheck.afp.com",
        'ar': "site:fatabyyano.net OR site:misbar.com OR site:factcheck.afp.com",
        'it': "site:facta.news OR site:lavoce.info OR site:pagellapolitica.it",
        'ja': "site:factcheck.jp OR site:afpbb.com",
        'ko': "site:factcheck.snu.ac.kr OR site:factcheck.kr"
    }
    
    fact_sites = sites.get(language, sites['en'])
    
    with DDGS() as ddgs:
        results = ddgs.text(f"{query} {fact_sites}", region=region, max_results=5)
        for r in results:
            search_results.append(f"[{r['title']}]({r['href']})")
    
    return "\n".join(search_results) if search_results else "No fact-checking results found."

# ✅ AI Misinformation Analysis
def analyze_misinformation(text, original_language='en'):
    prompt = f"""
    Analyze the following text (originally in {original_language}) and determine if it contains misinformation:
    {text}
    
    Please consider:
    1. Factual accuracy
    2. Claims that can be verified
    3. Logical consistency
    4. Cultural or regional context
    5. Known patterns of misinformation
    
    Provide a detailed analysis with confidence score.
    """
    response = llm.predict(prompt)
    return response

# ✅ AI-Generated Image Detection
def detect_ai_generated(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return "AI-Generated Image Detected" if laplacian < 50 else "Likely a Real Image"

# ✅ Summarize Findings with Native Language Output
def summarize_findings(text_analysis, fact_check_results, image_analysis, language_code, language_name):
    prompt = f"""
    Given the following analysis for content originally in {language_name}, summarize if the content is real or fake:
    - Text Analysis: {text_analysis}
    - Fact Check Results: {fact_check_results}
    - Image Analysis: {image_analysis}
    
    Provide a clear and concise decision with supporting evidence.
    """
    # Get analysis in English first
    english_response = llm.predict(prompt)
    
    # If language is not English, translate back to original language
    if language_code != 'en':
        return translate_text(english_response, 'en', language_code)
    else:
        return english_response

# ✅ Search News Articles with Region Support
def search_news(query, language='en', max_results=5):
    region_map = {'en': 'us', 'es': 'es', 'fr': 'fr', 'de': 'de', 'ar': 'ae', 
                 'zh-cn': 'cn', 'hi': 'in', 'ru': 'ru', 'pt': 'br', 'it': 'it',
                 'ja': 'jp', 'ko': 'kr'}
    region = region_map.get(language, 'us')
    
    with DDGS() as ddgs:
        return [{"title": r["title"], "href": r["href"]} for r in ddgs.text(query, region=region, max_results=max_results)]

# ✅ Verify News Using AI with Native Language Output
def verify_news(content, search_results, language_code, language_name):
    context = "\n".join([f"- {r['title']} ({r['href']})" for r in search_results])
    
    prompt = f"""
    You are an AI fact-checking assistant. Analyze the given news content (originally in {language_name}) and compare it with trusted sources below.
    
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
    # Get analysis in English
    english_response = llm.invoke([HumanMessage(content=prompt)]).content
    
    # If language is not English, translate back to original language
    if language_code != 'en':
        return translate_text(english_response, 'en', language_code)
    else:
        return english_response

# ✅ Streamlit UI with Sidebar
st.title("🌎 Multilingual Fake News & AI Image Detector")

# Language mapping for UI to code
language_map = {
    'English': 'en', 'Spanish': 'es', 'French': 'fr', 
    'German': 'de', 'Chinese': 'ch_sim', 'Arabic': 'ar',
    'Hindi': 'hi', 'Russian': 'ru', 'Portuguese': 'pt',
    'Italian': 'it', 'Japanese': 'ja', 'Korean': 'ko'
}

# Language mapping from detection codes to EasyOCR codes
langdetect_to_easyocr = {
    'en': 'en', 'es': 'es', 'fr': 'fr', 
    'de': 'de', 'zh-cn': 'ch_sim', 'ar': 'ar',
    'hi': 'hi', 'ru': 'ru', 'pt': 'pt',
    'it': 'it', 'ja': 'ja', 'ko': 'ko'
}

option = st.sidebar.radio("Select Detection Mode", ["Post Detection", "News Detection"])

# Language settings in sidebar
st.sidebar.header("🌐 Language Settings")
language_options = list(language_map.keys())
auto_detect = st.sidebar.checkbox("Auto-detect language", value=True)
return_native = st.sidebar.checkbox("Return results in original language", value=True)

if not auto_detect:
    selected_languages = st.sidebar.multiselect("Select languages for detection:", language_options, default=['English'])
    language_codes = [language_map[lang] for lang in selected_languages]
else:
    selected_languages = ['English']  # Default
    language_codes = ['en']  # Default

if option == "Post Detection":
    st.header("📸 Post Detection")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extracting text with a spinner
        with st.spinner("🔍 Extracting text from image..."):
            # First pass with default or selected language
            initial_extracted_text = extract_text("uploaded_image.jpg", languages=language_codes)
            
            if auto_detect and initial_extracted_text.strip():
                # Detect language from extracted text
                detected_code, detected_lang = detect_language(initial_extracted_text)
                st.info(f"🔍 Detected language: {detected_lang}")
                
                # Re-extract with detected language if different
                if detected_code != 'en' and detected_code in langdetect_to_easyocr:
                    easyocr_lang = langdetect_to_easyocr[detected_code]
                    extracted_text = extract_text("uploaded_image.jpg", languages=[easyocr_lang])
                    original_language_code = detected_code
                    original_language_name = detected_lang
                else:
                    extracted_text = initial_extracted_text
                    original_language_code = 'en'
                    original_language_name = 'English'
            else:
                extracted_text = initial_extracted_text
                original_language_code = language_codes[0] if language_codes else 'en'
                original_language_name = selected_languages[0] if selected_languages else 'English'
                
        st.subheader("📝 Extracted Text:")
        st.write(extracted_text)
        
        # Translate if not English
        if original_language_code != 'en':
            with st.spinner("🔄 Translating to English for analysis..."):
                translated_text = translate_text(extracted_text, original_language_code, 'en')
                st.subheader("🔄 English Translation:")
                st.write(translated_text)
        else:
            translated_text = extracted_text

        # Fact-checking with a spinner
        with st.spinner("🌐 Fact-checking with regional sources..."):
            fact_results = search_fact_check(translated_text, original_language_code)
        st.subheader("🔍 Fact-Check Results:")
        st.write(fact_results)

        # AI Misinformation Analysis with a spinner
        with st.spinner("🧠 Analyzing misinformation..."):
            ai_analysis = analyze_misinformation(translated_text, original_language_name)
        st.subheader("🧠 AI Analysis:")
        st.write(ai_analysis)

        # AI-Generated Image Detection with a spinner
        with st.spinner("🎭 Checking for AI-generated images..."):
            image_result = detect_ai_generated("uploaded_image.jpg")
        st.subheader("🎭 Image Analysis:")
        st.write(image_result)

        # Summarizing the findings with a spinner
        with st.spinner("📊 Summarizing findings..."):
            final_decision = summarize_findings(
                ai_analysis, 
                fact_results, 
                image_result,
                original_language_code if return_native else 'en',
                original_language_name
            )
        st.subheader("📢 Final Decision:")
        if original_language_code != 'en' and return_native:
            st.info(f"Results provided in original language: {original_language_name}")
        st.markdown(final_decision)

elif option == "News Detection":
    st.header("📰 News Detection")
    news_input = st.text_area("Paste the news content or claim:")
    
    if news_input.strip():
        # Auto-detect language if enabled
        if auto_detect:
            detected_code, detected_lang = detect_language(news_input)
            st.info(f"🔍 Detected language: {detected_lang}")
            language_code = detected_code
            language_name = detected_lang
        else:
            language_code = language_codes[0] if language_codes else 'en'
            language_name = selected_languages[0] if selected_languages else 'English'
        
        # Store original text
        original_text = news_input
        
        # Translate if not English
        if language_code != 'en':
            with st.spinner("🔄 Translating to English for analysis..."):
                translated_news = translate_text(news_input, language_code, 'en')
                if not return_native:  # Only show translation if not returning in native language
                    st.subheader("🔄 English Translation:")
                    st.write(translated_news)
                query_text = translated_news
        else:
            translated_news = news_input
            query_text = news_input
    
    if st.button("Authenticate"):
        if news_input.strip():
            with st.spinner("🔍 Searching for related news..."):
                search_results = search_news(query_text, language=language_code)

            if not search_results:
                error_msg = "❌ No relevant news found! Unable to verify."
                if language_code != 'en' and return_native:
                    error_msg = translate_text(error_msg, 'en', language_code)
                st.error(error_msg)
            else:
                result_msg = "✅ Relevant news articles found:"
                if language_code != 'en' and return_native:
                    result_msg = translate_text(result_msg, 'en', language_code)
                st.write(result_msg)
                
                for idx, article in enumerate(search_results, 1):
                    st.markdown(f"**{idx}. [{article['title']}]({article['href']})**")

                with st.spinner("🤖 Analyzing with AI..."):
                    verification_result = verify_news(
                        translated_news, 
                        search_results,
                        language_code if return_native else 'en',
                        language_name
                    )

                result_header = "📢 Fake News Detection Result:"
                if language_code != 'en' and return_native:
                    result_header = translate_text(result_header, 'en', language_code)
                    st.info(f"Results provided in original language: {language_name}")
                
                st.subheader(result_header)
                st.markdown(verification_result)
        else:
            warning_msg = "⚠️ Please enter news content to verify."
            st.warning(warning_msg)

# Additional information in sidebar
st.sidebar.subheader("Supported Languages")
st.sidebar.info(", ".join(language_options))


