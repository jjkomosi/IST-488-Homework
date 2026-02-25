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
# ─── Build Vector DB ──────────────────────────────────────────────────────────
# We use a new collection name to force a rebuild with our improved parser.
# The HW4 collection only had JSON data; this one also scrapes the HTML body
# for contact info, meeting details, officers, and descriptions.
chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW5')
collection = chroma_client.get_or_create_collection('HW5Collection')

# ─── Initialize API Clients ──────────────────────────────────────────────────
if 'anthropic_client' not in st.session_state:
    st.session_state.anthropic_client = anthropic.Anthropic(
        api_key=st.secrets["ANTHROPIC_API_KEY"]
    )

if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# ─── HTML Parsing ─────────────────────────────────────────────────────────────
def extract_org_info(file_path):
    """
    Extract structured organization info from CampusLabs HTML files.

    We pull from TWO sources to get complete data:
      1. The JSON blob in window.initialAppState (name, summary, status, etc.)
      2. The rendered HTML body (description, meeting info, officers, contact
         details) — much of this data is ONLY in the HTML, not in the JSON.
         For example, description is often null in the JSON but fully present
         in the rendered page.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # ── Source 1: JSON blob ──────────────────────────────────────────────
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

    name = org.get('name', 'Unknown Organization')
    summary = org.get('summary', '')
    json_description = org.get('description', '')
    status = org.get('status', '')
    email = org.get('email', '')
    start_date = org.get('startDate', '')
    org_type = org.get('organizationType', {}).get('name', '')

    if json_description:
        desc_soup = BeautifulSoup(json_description, 'html.parser')
        json_description = desc_soup.get_text(strip=True)

    # Primary contact from JSON
    contact = org.get('primaryContact', {})
    contact_name = ''
    contact_email = ''
    if contact:
        first = contact.get('preferredFirstName') or contact.get('firstName', '')
        last = contact.get('lastName', '')
        contact_name = f"{first} {last}".strip()
        contact_email = contact.get('primaryEmailAddress', '')

    # Contact info (address, phone) from JSON
    contact_info_list = org.get('contactInfo', [])
    address = ''
    phone = ''
    if contact_info_list:
        ci = contact_info_list[0]
        street = ci.get('street1', '')
        city = ci.get('city', '')
        state = ci.get('state', '')
        zipcode = ci.get('zip', '')
        if street:
            address = f"{street}, {city}, {state} {zipcode}".strip(', ')
        phone = ci.get('phoneNumber', '')

    # Social media from JSON
    social = org.get('socialMedia', {})
    social_links = []
    for platform, url in social.items():
        if url and isinstance(url, str) and url.strip():
            social_links.append(f"{platform}: {url}")

    # ── Source 2: Rendered HTML body ─────────────────────────────────────
    # The "Additional Information" section contains key-value pairs like
    # Description, Website, Meeting Day, Meeting Time, President Name, etc.
    # These are structured as pairs of divs: a bold label div + a value div.
    soup = BeautifulSoup(html_content, 'html.parser')

    html_fields = {}
    bold_divs = soup.find_all('div', style=lambda s: s and 'font-weight: bold' in s)
    for bold_div in bold_divs:
        label = bold_div.get_text(strip=True).rstrip(':')
        # The value is in the next sibling div
        value_div = bold_div.find_next_sibling('div')
        if value_div:
            value = value_div.get_text(strip=True)
            if value:
                html_fields[label] = value

    # Extract officer names from the roster cards in the HTML
    officers = []
    officer_cards = soup.find_all('div', style=lambda s: s and 'text-align: center' in str(s) and 'min-height: 200px' in str(s))
    for card in officer_cards:
        role_div = card.find('div', style=lambda s: s and 'font-weight: bold' in str(s))
        name_div = card.find('div', style=lambda s: s and 'margin: 5px 0px' in str(s))
        if role_div and name_div:
            role = role_div.get_text(strip=True)
            officer_name = name_div.get_text(strip=True)
            officers.append(f"{role}: {officer_name}")

    # ── Assemble full text ───────────────────────────────────────────────
    # Use HTML-sourced description if JSON description is empty
    html_description = html_fields.get('Description', '')
    description = json_description or html_description

    parts = [f"Organization: {name}"]
    if org_type:
        parts.append(f"Type: {org_type}")
    if status:
        parts.append(f"Status: {status}")
    if summary:
        parts.append(f"Summary: {summary}")
    if description and description != summary:
        parts.append(f"Description: {description}")

    # Contact information block
    if contact_name:
        parts.append(f"Primary Contact: {contact_name} ({contact_email})")
    if email:
        parts.append(f"Email: {email}")
    if address:
        parts.append(f"Address: {address}")
    if phone:
        parts.append(f"Phone: {phone}")

    # Meeting information from HTML
    meeting_day = html_fields.get('Meeting Day', '')
    meeting_time = html_fields.get('Meeting time', '')
    meeting_ampm = html_fields.get('AM/PM', '')
    meeting_location = html_fields.get('Meeting Location', '')
    if meeting_day:
        meeting_str = f"Meeting: {meeting_day}"
        if meeting_time:
            meeting_str += f" at {meeting_time}"
            if meeting_ampm:
                meeting_str += f" {meeting_ampm}"
        if meeting_location:
            meeting_str += f", Location: {meeting_location}"
        parts.append(meeting_str)

    # Leadership from HTML fields (President, VP, etc.)
    for label_key in ['President Name', 'Vice-President Name',
                      'Secretary (or other eboard position) name',
                      'Treasurer/Fiscal Agent Name',
                      'Full-time SU/ESF Faculty/Staff Advisor name']:
        if label_key in html_fields:
            parts.append(f"{label_key}: {html_fields[label_key]}")

    # Website from HTML
    website = html_fields.get('Website', '')
    if website:
        parts.append(f"Website: {website}")

    # Membership process from HTML
    member_process = html_fields.get(
        'Member/Selection Process:\n \n Please describe the process students should take to join your organization.',
        ''
    )
    if not member_process:
        # Try a simpler match
        for key, val in html_fields.items():
            if 'member' in key.lower() or 'selection process' in key.lower():
                member_process = val
                break
    if member_process:
        parts.append(f"How to Join: {member_process}")

    # Officers from roster cards
    if officers:
        parts.append("Officers: " + "; ".join(officers))

    # Social media
    if social_links:
        parts.append("Social Media: " + "; ".join(social_links))
    if start_date:
        parts.append(f"Start Date: {start_date[:10]}")

    return {
        'name': name,
        'full_text': "\n".join(parts)
    }


# ─── Chunking ────────────────────────────────────────────────────────────────
# CHUNKING METHOD: Newline-based logical splitting
#
# With the improved parser, each org's text now contains structured fields
# separated by newlines (name, type, summary, description, contact info,
# meeting info, officers, etc.). We split on newline boundaries into chunks
# of roughly equal size. This keeps related fields together while giving
# the retriever more granular matching — e.g., one chunk may match on
# the org description while another matches on contact/meeting info.

def chunk_text(text):
    """Split text into 2-3 roughly equal chunks at newline boundaries."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    if len(lines) <= 2:
        return [text]

    # For longer extractions, split into 3 chunks; otherwise 2
    num_chunks = 3 if len(lines) >= 9 else 2
    chunk_size = max(1, len(lines) // num_chunks)

    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk = '\n'.join(lines[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)

    # If we ended up with too many small chunks, merge the last two
    if len(chunks) > num_chunks:
        chunks[-2] = chunks[-2] + '\n' + chunks[-1]
        chunks.pop()

    return chunks if chunks else [text]


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
                # Prepend org name to every chunk so embeddings always
                # associate contact info, officers, etc. with the right org
                if not chunk.startswith(f"Organization: {org_info['name']}"):
                    chunk = f"Organization: {org_info['name']}\n{chunk}"
                doc_id = f"{html_file}_chunk{i}"
                add_to_collection(collection, chunk, doc_id)

        progress.progress((idx + 1) / total, text=f"Processing {idx + 1}/{total} files...")

    progress.empty()


# Build the vector DB only if it doesn't already exist
if collection.count() == 0:
    load_html_to_collection('./HW4_Data/', collection)


# ─── Tool Definition for Function Calling ─────────────────────────────────────
# ENHANCEMENT OVER HW4:
# Instead of always embedding the raw user prompt and querying ChromaDB,
# we define a tool that the LLM can *choose* to call. This way:
#   1. The LLM decides IF a search is needed (skips greetings, follow-ups, etc.)
#   2. The LLM crafts an optimized search query (better than raw user text)
#   3. The LLM sees the results as a tool response and synthesizes an answer
#
# This is the "show the LLM the function call result directly" approach from
# the assignment — it's more elegant because the tool use flow is native to
# the Anthropic API and keeps the system prompt clean and static.

TOOLS = [
    {
        "name": "relevant_club_info",
        "description": (
            "Search the Syracuse University student organization database for "
            "relevant club information. Use this tool whenever the user asks about "
            "student organizations, clubs, groups, or activities at Syracuse University. "
            "Craft a focused search query that captures the key topic the user is asking about."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A search query to find relevant student organizations. "
                        "Should be a concise phrase capturing what the user wants to know, "
                        "e.g. 'computer science technology clubs', 'dance performing arts', "
                        "'community service volunteering organizations'."
                    )
                }
            },
            "required": ["query"]
        }
    }
]


def relevant_club_info(query):
    """
    Takes a search query (from the LLM), embeds it, queries ChromaDB,
    and returns the top matching organization chunks as a formatted string.
    """
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    if not results['documents'][0]:
        return "No relevant student organizations found for this query."

    output_parts = []
    for i in range(len(results['documents'][0])):
        doc_text = results['documents'][0][i]
        doc_id = results['ids'][0][i]
        output_parts.append(f"--- Source: {doc_id} ---\n{doc_text}")

    return "\n\n".join(output_parts)


def process_tool_call(tool_name, tool_input):
    """Route tool calls to the appropriate function."""
    if tool_name == "relevant_club_info":
        return relevant_club_info(tool_input["query"])
    return "Unknown tool."


# ─── Agentic Chat Loop ───────────────────────────────────────────────────────
def get_response(user_input):
    """
    Run the agentic tool-use loop:
      1. Send user message + tool definitions to Claude
      2. If Claude calls relevant_club_info, execute it and return results
      3. Claude sees the results and generates a final answer
      4. If Claude doesn't call a tool, it answers directly (e.g. for greetings)
    """
    client = st.session_state.anthropic_client

    system_prompt = """You are a helpful Syracuse University student organization assistant.
You have access to a tool called relevant_club_info that searches a database of 
student organizations at Syracuse University. 

Use this tool when the user asks about clubs, organizations, groups, or activities.
Do NOT use the tool for greetings, thank-yous, or general conversation.

When you get results from the tool, clearly state which organization(s) you are 
referencing. If the results don't contain relevant information, say so honestly 
and suggest the user try rephrasing their question."""

    # Build the 5-message conversation buffer (last 5 exchanges = 10 messages)
    buffer_messages = st.session_state.messages[-10:]
    api_messages = [{"role": m["role"], "content": m["content"]} for m in buffer_messages]

    # First API call — Claude may or may not call the tool
    response = client.messages.create(
        model='claude-haiku-4-5-20251001',
        max_tokens=1024,
        system=system_prompt,
        tools=TOOLS,
        messages=api_messages
    )

    # Agentic loop: keep going while Claude wants to use tools
    while response.stop_reason == "tool_use":
        # Find the tool use block in the response
        tool_use_block = next(
            block for block in response.content if block.type == "tool_use"
        )

        tool_name = tool_use_block.name
        tool_input = tool_use_block.input

        # Execute the tool
        tool_result = process_tool_call(tool_name, tool_input)

        # Append assistant's tool call and the tool result to the conversation
        api_messages.append({"role": "assistant", "content": response.content})
        api_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_block.id,
                    "content": tool_result
                }
            ]
        })

        # Second API call — Claude now has the search results and generates an answer
        response = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=1024,
            system=system_prompt,
            tools=TOOLS,
            messages=api_messages
        )

    # Extract the final text response
    final_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            final_text += block.text

    return final_text


# ─── Streamlit UI ─────────────────────────────────────────────────────────────
st.title("HW5: Syracuse Student Organization Chatbot")
st.caption("Ask me anything about student organizations at Syracuse University!")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display full chat history
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

    # Get response using the agentic tool-use loop
    with st.spinner("Thinking..."):
        assistant_message = get_response(user_input)

    # Display assistant response
    with st.chat_message('assistant'):
        st.markdown(assistant_message)
    st.session_state.messages.append({'role': 'assistant', 'content': assistant_message})