import streamlit as st
from openai import OpenAI
import tiktoken
from anthropic import Anthropic
import requests
from bs4 import BeautifulSoup

# ============================================
# HELPER FUNCTION: Read URL Content
# ============================================
def read_url_content(url):
    """Read content from a URL and return text."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

# ============================================
# TOKEN COUNTING FUNCTIONS
# ============================================
def count_tokens(text, model="gpt-5"):
    """Count tokens in a string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def count_messages_tokens(messages, model="gpt-5"):
    """Count total tokens across all messages."""
    total = 0
    for message in messages:
        total += 4  # overhead per message
        total += count_tokens(message["content"], model)
    total += 2  # conversation overhead
    return total

def apply_token_buffer(messages, max_tokens, model="gpt-5"):
    """
    Trim conversation history to stay under token limit.
    Removes oldest messages first, preserves system message.
    NOTE: Only counts conversational tokens (user/assistant), not system prompt.
    """
    if len(messages) <= 1:
        return messages
    
    # Preserve system message if present
    system_msg = None
    if messages[0]["role"] == "system":
        system_msg = messages[0]
        working_messages = messages[1:]
    else:
        working_messages = messages[:]
    
    # Count only conversational tokens (exclude system message)
    current_tokens = count_messages_tokens(working_messages, model)
    
    # Remove oldest messages until under limit
    while current_tokens > max_tokens and len(working_messages) > 1:
        working_messages.pop(0)
        current_tokens = count_messages_tokens(working_messages, model)
    
    if system_msg:
        return [system_msg] + working_messages
    return working_messages

# ============================================
# APP SETUP
# ============================================
st.title("Jonah's HW 3: URL-Based Q/A Chatbot")

# Description at the top
st.write("""
### How this chatbot works:
This chatbot answers questions based on content from up to two URLs you provide. 

**Conversation Memory:** This implementation uses a **token buffer of 2,000 tokens for conversation history**. 
The system prompt (containing the URL content) is always preserved and never discarded. The token 
buffer only counts conversational messages (user questions and assistant responses), and older 
conversation pairs are removed as needed to stay within the 2,000 token limit for the conversation.

**To use:** Enter your URL(s) in the sidebar, select your preferred LLM, then ask questions!
""")

# ============================================
# SIDEBAR: Configuration Options
# ============================================
st.sidebar.header('⚙️ Configuration')

# URL inputs
st.sidebar.subheader('📄 Document URLs')
url1 = st.sidebar.text_input('URL 1 (optional):', value='')
url2 = st.sidebar.text_input('URL 2 (optional):', value='')

# LLM Selection
st.sidebar.subheader('🤖 LLM Selection')
llm_provider = st.sidebar.selectbox(
    'Choose LLM provider:',
    ('OpenAI (GPT-5)', 'Anthropic (Claude Sonnet 4.5)')
)

# Set model based on selection
if llm_provider == 'OpenAI (GPT-5)':
    model_to_use = "gpt-5"
    use_anthropic = False
else:
    model_to_use = "claude-sonnet-4-5-20250929"
    use_anthropic = True

# Buffer configuration
MAX_CONTEXT_TOKENS = 2000

# ============================================
# INITIALIZE CLIENTS
# ============================================
if 'openai_client' not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.openai_client = OpenAI(api_key=api_key)

if 'anthropic_client' not in st.session_state:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
    st.session_state.anthropic_client = Anthropic(api_key=api_key)

# ============================================
# LOAD URL CONTENT
# ============================================
if 'url_content_loaded' not in st.session_state:
    st.session_state.url_content_loaded = False

if 'url_content' not in st.session_state:
    st.session_state.url_content = ""

# Load URLs button
if st.sidebar.button('Load URL Content'):
    with st.spinner('Loading URL content...'):
        content_parts = []
        
        if url1:
            content1 = read_url_content(url1)
            if content1:
                content_parts.append(f"=== Content from URL 1 ({url1}) ===\n{content1}")
                st.sidebar.success(f"✓ Loaded URL 1 ({len(content1)} chars)")
        
        if url2:
            content2 = read_url_content(url2)
            if content2:
                content_parts.append(f"\n\n=== Content from URL 2 ({url2}) ===\n{content2}")
                st.sidebar.success(f"✓ Loaded URL 2 ({len(content2)} chars)")
        
        if content_parts:
            st.session_state.url_content = "\n\n".join(content_parts)
            st.session_state.url_content_loaded = True
            st.sidebar.success('✓ All URLs loaded successfully!')
        else:
            st.sidebar.error('No content could be loaded from the URLs')

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================
# SYSTEM PROMPT WITH URL CONTENT
# ============================================
def get_system_prompt():
    """Create system prompt with URL content."""
    if st.session_state.url_content_loaded and st.session_state.url_content:
        return f"""You are a helpful assistant that answers questions based on the following document content.
Use the information from these documents to provide accurate, detailed answers.
If the answer is not in the documents, say so.

{st.session_state.url_content}"""
    else:
        return "You are a helpful assistant."

# ============================================
# HELPER FUNCTION: Get Response
# ============================================
def get_response_openai(user_message):
    """Get response from OpenAI with token buffer."""
    client = st.session_state.openai_client
    
    # Build messages with system prompt
    messages_to_send = [{"role": "system", "content": get_system_prompt()}]
    
    # Add conversation history
    messages_to_send.extend(st.session_state.messages)
    
    # Add current user prompt
    messages_to_send.append({"role": "user", "content": user_message})
    
    # Apply token buffer
    messages_to_send = apply_token_buffer(
        messages_to_send,
        max_tokens=MAX_CONTEXT_TOKENS,
        model=model_to_use
    )
    
    # Debug info - count only conversational tokens
    conversational_messages = [msg for msg in messages_to_send if msg["role"] != "system"]
    conversational_tokens = count_messages_tokens(conversational_messages, model_to_use)
    st.sidebar.write(f"📊 Conversation messages: {len(conversational_messages)}")
    st.sidebar.write(f"📊 Conversation tokens: {conversational_tokens}/{MAX_CONTEXT_TOKENS}")
    
    response = client.chat.completions.create(
        model=model_to_use,
        messages=messages_to_send,
        stream=True
    )
    
    return response

def get_response_anthropic(user_message):
    """Get response from Anthropic with token buffer."""
    client = st.session_state.anthropic_client
    
    # System prompt (separate for Anthropic)
    system_prompt = get_system_prompt()
    
    # Build messages (no system role in messages for Anthropic)
    messages_to_send = []
    messages_to_send.extend(st.session_state.messages)
    messages_to_send.append({"role": "user", "content": user_message})
    
    # For Anthropic, we need to manage tokens differently
    # System prompt is separate, only count conversational messages
    available_tokens = MAX_CONTEXT_TOKENS
    
    # Trim conversation history to fit
    while len(messages_to_send) > 0:
        current_tokens = count_messages_tokens(messages_to_send, "gpt-5")
        if current_tokens <= available_tokens:
            break
        if len(messages_to_send) > 1:
            messages_to_send.pop(0)
        else:
            break
    
    conversational_tokens = count_messages_tokens(messages_to_send, 'gpt-5')
    st.sidebar.write(f"📊 Conversation messages: {len(messages_to_send)}")
    st.sidebar.write(f"📊 Conversation tokens: {conversational_tokens}/{MAX_CONTEXT_TOKENS}")
    
    # Create streaming response
    with client.messages.stream(
        model=model_to_use,
        max_tokens=1024,
        system=system_prompt,
        messages=messages_to_send
    ) as stream:
        for text in stream.text_stream:
            yield text

# ============================================
# DISPLAY CONVERSATION HISTORY
# ============================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ============================================
# HANDLE USER INPUT
# ============================================
if prompt := st.chat_input("Ask a question..."):
    
    # Display user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get and display assistant's response
    with st.chat_message("assistant"):
        if use_anthropic:
            stream = get_response_anthropic(prompt)
            response_text = st.write_stream(stream)
        else:
            stream = get_response_openai(prompt)
            response_text = st.write_stream(stream)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})

# ============================================
# SIDEBAR: Additional Info
# ============================================
st.sidebar.write("---")
st.sidebar.write("**Session Info:**")
st.sidebar.write(f"Conversation messages: {len(st.session_state.messages)}")
st.sidebar.write(f"URLs loaded: {'Yes' if st.session_state.url_content_loaded else 'No'}")
st.sidebar.write(f"Current model: {model_to_use}")