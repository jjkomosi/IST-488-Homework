import streamlit as st
import sys

# sqlite3 fix - MUST be before chromadb import
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import anthropic
from openai import OpenAI
from bs4 import BeautifulSoup
from pathlib import Path
import json
import re
import os

# ─── ChromaDB Setup ───────────────────────────────────────────────────────────
# Create ChromaDB client - persistent so we only build the DB once
chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW4')
collection = chroma_client.get_or_create_collection('HW4Collection')

# ─── Initialize API Clients ──────────────────────────────────────────────────
if 'anthropic_client' not in st.session_state:
    st.session_state.anthropic_client = anthropic.Anthropic(
        api_key=st.secrets["ANTHROPIC_API_KEY"]
    )

# OpenAI client for embeddings (reusing from Lab 4 pattern)
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# ─── HTML Parsing ─────────────────────────────────────────────────────────────
def extract_org_info(file_path):
    """
    Extract structured organization info from the CampusLabs HTML files.
    The actual data lives in a JSON blob inside a <script> tag
    (window.initialAppState), so we parse that rather than scraping
    visible HTML text, which is mostly empty template markup.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Extract the JSON from window.initialAppState
    match = re.search(r'window\.initialAppState\s*=\s*({.*?});\s*</script>', html_content, re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None

    org = data.get('preFetchedData', {}).get('organization')
    if not org:
        return None

    # Build a clean text representation of the organization
    name = org.get('name', 'Unknown Organization')
    summary = org.get('summary', '')
    description = org.get('description', '')
    status = org.get('status', '')
    email = org.get('email', '')
    start_date = org.get('startDate', '')
    org_type = org.get('organizationType', {}).get('name', '')

    # Clean HTML tags from description
    if description:
        desc_soup = BeautifulSoup(description, 'html.parser')
        description = desc_soup.get_text(strip=True)

    # Primary contact
    contact = org.get('primaryContact', {})
    contact_name = ''
    if contact:
        first = contact.get('preferredFirstName') or contact.get('firstName', '')
        last = contact.get('lastName', '')
        contact_name = f"{first} {last}".strip()
        contact_email = contact.get('primaryEmailAddress', '')

    # Social media
    social = org.get('socialMedia', {})
    social_links = []
    for platform, url in social.items():
        if url and isinstance(url, str) and url.strip():
            social_links.append(f"{platform}: {url}")

    # Assemble full text
    parts = [f"Organization: {name}"]
    if org_type:
        parts.append(f"Type: {org_type}")
    if status:
        parts.append(f"Status: {status}")
    if summary:
        parts.append(f"Summary: {summary}")
    if description and description != summary:
        parts.append(f"Description: {description}")
    if contact_name:
        parts.append(f"Primary Contact: {contact_name} ({contact_email})")
    elif email:
        parts.append(f"Email: {email}")
    if social_links:
        parts.append("Social Media: " + "; ".join(social_links))
    if start_date:
        parts.append(f"Start Date: {start_date[:10]}")

    return {
        'name': name,
        'full_text': "\n".join(parts)
    }


# ─── Chunking ────────────────────────────────────────────────────────────────
# CHUNKING METHOD: Midpoint sentence-boundary splitting
#
# Each HTML file represents a single student organization page from CampusLabs.
# The extracted text is relatively short (typically a few sentences of structured
# info: name, type, summary, description, contact, social media links).
#
# We split each document into two chunks at the sentence midpoint. This is the
# most appropriate method here because:
#   1. The documents are already short and topically focused (one org each).
#   2. More complex methods (semantic, recursive, token-based) would add
#      unnecessary overhead for documents this small.
#   3. Splitting at a sentence boundary keeps each chunk coherent and readable.
#   4. Two chunks per document satisfies the assignment requirement while
#      still giving the retriever more granular matching capability —
#      e.g., one chunk may match on the org description while the other
#      matches on contact/social info.

def chunk_text(text):
    """Split text into two roughly equal chunks at a sentence boundary."""
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)

    if len(sentences) <= 1:
        # If we can't split by sentences, split by newlines
        sentences = text.split('\n')

    if len(sentences) <= 1:
        # Last resort: split at character midpoint
        mid = len(text) // 2
        return [text[:mid], text[mid:]]

    mid = len(sentences) // 2
    chunk1 = ' '.join(sentences[:mid]).strip()
    chunk2 = ' '.join(sentences[mid:]).strip()

    # Make sure neither chunk is empty
    if not chunk1 or not chunk2:
        return [text]

    return [chunk1, chunk2]


# ─── Embedding + Collection Loading ──────────────────────────────────────────
def get_embedding(text):
    """Get embedding vector from OpenAI."""
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=text,
        model='text-embedding-3-small'
    )
    return response.data[0].embedding


def add_to_collection(collection, text, doc_id):
    """Add a single document with its embedding to ChromaDB."""
    embedding = get_embedding(text)
    collection.add(
        documents=[text],
        ids=[doc_id],
        embeddings=[embedding]
    )


def load_html_to_collection(folder_path, collection):
    """Load all HTML files, extract org info, chunk, and add to ChromaDB."""
    html_files = [f for f in os.listdir(folder_path) if f.endswith('.html')]

    progress = st.progress(0, text="Building vector database...")
    total = len(html_files)

    for idx, html_file in enumerate(html_files):
        file_path = os.path.join(folder_path, html_file)
        org_info = extract_org_info(file_path)

        if org_info:
            chunks = chunk_text(org_info['full_text'])
            for i, chunk in enumerate(chunks):
                doc_id = f"{html_file}_chunk{i}"
                add_to_collection(collection, chunk, doc_id)

        progress.progress((idx + 1) / total, text=f"Processing {idx + 1}/{total} files...")

    progress.empty()


# Build the vector DB only if it doesn't already exist
if collection.count() == 0:
    load_html_to_collection('./HW4_Data/', collection)


# ─── Streamlit UI ─────────────────────────────────────────────────────────────
st.title("HW4: Syracuse Student Organization Chatbot")
st.caption("Ask me anything about student organizations at Syracuse University!")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display full chat history in the UI (so user can scroll back)
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Chat input
user_input = st.chat_input("Ask about student organizations at Syracuse...")

if user_input:
    # Display user message
    with st.chat_message('user'):
        st.markdown(user_input)
    st.session_state.messages.append({'role': 'user', 'content': user_input})

    # Step 1: Embed the user's question and query ChromaDB
    query_embedding = get_embedding(user_input)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    # Step 2: Build RAG context from retrieved documents
    context = ""
    sources = []
    for i in range(len(results['documents'][0])):
        doc_text = results['documents'][0][i]
        doc_id = results['ids'][0][i]
        sources.append(doc_id)
        context += f"\n--- {doc_id} ---\n{doc_text}\n"

    # Step 3: Build the 5-message conversation buffer
    # We keep the LAST 5 exchanges (10 messages: 5 user + 5 assistant)
    # This is sent to the LLM so it has conversational context,
    # while the full history is still shown in the UI.
    buffer_messages = st.session_state.messages[-10:]

    # Convert to Anthropic message format
    api_messages = []
    for msg in buffer_messages:
        api_messages.append({
            'role': msg['role'],
            'content': msg['content']
        })

    system_prompt = f"""You are a helpful Syracuse University student organization assistant.
Answer questions about student organizations at Syracuse University using the retrieved 
organization data below.

When answering, clearly state which organization(s) you are referencing. If the retrieved
data doesn't contain relevant information, say so honestly and suggest the user try
rephrasing their question.

Retrieved organization data:
{context}
"""

    # Step 4: Call Claude Haiku via the Anthropic API
    client = st.session_state.anthropic_client
    response = client.messages.create(
        model='claude-haiku-4-5-20251001',
        max_tokens=1024,
        system=system_prompt,
        messages=api_messages
    )

    assistant_message = response.content[0].text

    # Display assistant response
    with st.chat_message('assistant'):
        st.markdown(assistant_message)
    st.session_state.messages.append({'role': 'assistant', 'content': assistant_message})