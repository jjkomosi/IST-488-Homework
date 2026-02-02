import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import requests
from bs4 import BeautifulSoup

def read_url_content(url):
    """Read content from a URL and return text."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

# --- SIDEBAR ---
st.sidebar.title(':green[URL Summarizer]')

# Summary type selection
st.sidebar.header(':green[Summary Options]')
summary_type = st.sidebar.selectbox(
    'Choose a summary format',
    ('100 words', '2 connecting paragraphs', '5 bullet points')
)

# Language selection
st.sidebar.header(':green[Output Language]')
output_language = st.sidebar.selectbox(
    'Choose output language',
    ('English', 'French', 'Spanish', 'German', 'Chinese')
)

# LLM selection
st.sidebar.header(':green[LLM Selection]')
llm_provider = st.sidebar.selectbox(
    'Choose LLM provider',
    ('OpenAI', 'Anthropic (Claude)')
)

# Advanced model checkbox
use_advanced_model = st.sidebar.checkbox('Use advanced model')

# Set model names based on selections
if llm_provider == "OpenAI":
    if use_advanced_model:
        model_name = "gpt-5"
    else:
        model_name = "gpt-5-nano"
    st.sidebar.info(f"Using: {model_name}")
else:  # Anthropic
    if use_advanced_model:
        model_name = "claude-sonnet-4-5-20250929"
    else:
        model_name = "claude-haiku-4-5-20251001"
    st.sidebar.info(f"Using: {model_name}")

# Main page
st.title(':green[🌐 URL Summarizer]')
st.subheader(':green[HW 2: Multi-LLM Summary Generator]')

st.write("Enter a URL and get an AI-generated summary. Use the sidebar to customize your options.")

# URL input (at top of screen, not sidebar)
url_input = st.text_input("Enter a URL to summarize", placeholder="https://example.com")

# Get API keys from secrets
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]
except KeyError as e:
    st.error(f"API key not found: {e}. Please configure your API keys in secrets.")
    st.stop()

# Initialize the appropriate client
if llm_provider == "OpenAI":
    client = OpenAI(api_key=openai_api_key)
else:
    client = Anthropic(api_key=anthropic_api_key)

if url_input:
    if st.button("Generate Summary"):
        # Read content from URL
        with st.spinner("Reading URL content..."):
            url_content = read_url_content(url_input)
        
        if url_content is None:
            st.error("Failed to read URL content.")
            st.stop()
        
        # Build prompt based on summary type and language
        if summary_type == "100 words":
            summary_instruction = "Summarize the following content in exactly 100 words"
        elif summary_type == "2 connecting paragraphs":
            summary_instruction = "Summarize the following content in 2 connecting paragraphs"
        else:
            summary_instruction = "Summarize the following content in exactly 5 bullet points"
        
        prompt = f"{summary_instruction}. Output your response in {output_language}.\n\n{url_content}"
        
        # Generate summary based on selected LLM
        st.subheader(f':green[Summary ({summary_type}) in {output_language}]')
        
        if llm_provider == "OpenAI":
            messages = [{"role": "user", "content": prompt}]
            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
            )
            st.write_stream(stream)
        else:  # Anthropic
            with client.messages.stream(
                model=model_name,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                response_text = ""
                response_container = st.empty()
                for text in stream.text_stream:
                    response_text += text
                    response_container.write(response_text)