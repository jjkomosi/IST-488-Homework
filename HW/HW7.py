"""
HW7.py — News Information Bot
A RAG-powered news reporting bot for a global law firm to monitor client news.
Uses ChromaDB for retrieval and Claude's tool-use API for agentic query routing.
"""

import json
import time
import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import anthropic

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="News Intelligence Bot", page_icon="📰", layout="wide")

# ── Constants ────────────────────────────────────────────────────────
DB_DIR = "HW7_data/news_chroma_db"
COLLECTION_NAME = "news_articles"

MODELS = {
    "Claude Haiku (lower-cost)": "claude-haiku-4-5-20251001",
    "Claude Sonnet (higher-cost)": "claude-sonnet-4-5",
}

SYSTEM_PROMPT = """You are a news intelligence analyst for a large global law firm. 
Your job is to help attorneys and partners stay informed about news concerning the firm's clients.

You have access to a curated database of news articles. Use your tools to search and retrieve 
relevant articles before answering. Always ground your responses in the actual articles retrieved.

Guidelines:
- When asked to "find interesting news," use the rank_interesting_news tool and present a 
  numbered list with article titles, companies, dates, and a brief explanation of why each 
  is noteworthy (legal exposure, regulatory action, M&A, market impact, etc.).
- When asked about a specific company or topic, use the search_news tool.
- Always cite the source URL for each article you reference.
- Provide context: explain *why* a story matters to a law firm (litigation risk, regulatory 
  scrutiny, deal activity, reputational concerns, etc.).
- Be concise but thorough. Attorneys are busy.
- If the database has no relevant results, say so clearly.
"""

# ── Tool definitions for Claude ──────────────────────────────────────
TOOLS = [
    {
        "name": "search_news",
        "description": (
            "Search the news article database by topic, company name, or keyword. "
            "Returns the most semantically relevant articles. Use this when the user "
            "asks about a specific company, topic, event, or keyword."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query — a company name, topic, or keywords.",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to retrieve (default 10, max 20).",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "rank_interesting_news",
        "description": (
            "Retrieve a broad set of recent news articles for ranking by importance. "
            "Use this when the user asks for 'the most interesting news,' a general "
            "briefing, or a ranked overview of noteworthy stories. Optionally filter "
            "by topic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Optional topic to focus the ranking on (e.g., 'AI', 'regulation'). Leave empty for general.",
                    "default": "",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of articles to retrieve for ranking (default 20).",
                    "default": 20,
                },
            },
            "required": [],
        },
    },
]


# ── Load ChromaDB ────────────────────────────────────────────────────
@st.cache_resource
def load_collection():
    """Load the pre-built ChromaDB collection."""
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=st.secrets["OPENAI_API_KEY"],
        model_name="text-embedding-3-small",
    )
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )
    return collection


def execute_tool(tool_name: str, tool_input: dict, collection) -> str:
    """Execute a tool call and return the result as a string."""

    if tool_name == "search_news":
        query = tool_input["query"]
        n = min(tool_input.get("n_results", 10), 20)
        results = collection.query(query_texts=[query], n_results=n)
        return format_results(results)

    elif tool_name == "rank_interesting_news":
        topic = tool_input.get("topic", "")
        n = min(tool_input.get("n_results", 20), 30)
        # Broad semantic search — use topic if given, else a general query
        query = topic if topic else "important breaking news major development"
        results = collection.query(query_texts=[query], n_results=n)
        return format_results(results)

    return "Unknown tool."


def format_results(results: dict) -> str:
    """Format ChromaDB results into a readable string for the LLM."""
    if not results or not results["documents"] or not results["documents"][0]:
        return "No results found in the database."

    formatted = []
    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        formatted.append(
            f"[{i + 1}] Company: {meta.get('company', 'N/A')}\n"
            f"    Date: {meta.get('date', 'N/A')}\n"
            f"    URL: {meta.get('url', 'N/A')}\n"
            f"    Content: {doc}"
        )
    return "\n\n".join(formatted)


def run_agent(user_message: str, chat_history: list, model: str, collection) -> tuple:
    """
    Run the agentic tool-use loop.
    Returns (assistant_response_text, elapsed_seconds).
    """
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    # Build messages from chat history
    messages = []
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    start = time.time()

    # Agentic loop — keep going while the model wants to use tools
    while True:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Check if the model wants to use a tool
        if response.stop_reason == "tool_use":
            # There may be text + tool_use blocks in content
            assistant_content = response.content

            # Add assistant turn with full content (text + tool_use)
            messages.append({"role": "assistant", "content": assistant_content})

            # Process each tool use block
            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    result_str = execute_tool(block.name, block.input, collection)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        }
                    )

            # Add tool results as user turn
            messages.append({"role": "user", "content": tool_results})

        else:
            # Model is done — extract final text
            elapsed = time.time() - start
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            return final_text, elapsed


# ── Streamlit UI ─────────────────────────────────────────────────────
def main():
    st.title("📰 News Intelligence Bot")
    st.caption("RAG-powered news monitoring for a global law firm's client portfolio")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        selected_model = st.selectbox("Choose LLM", list(MODELS.keys()))
        model_id = MODELS[selected_model]

        st.divider()
        st.markdown("**Quick queries:**")
        if st.button("🔥 Most interesting news"):
            st.session_state["prefill"] = "Find the most interesting news"
        if st.button("🏦 JPMorgan Chase news"):
            st.session_state["prefill"] = "Find news about JPMorgan Chase"
        if st.button("🤖 AI & tech news"):
            st.session_state["prefill"] = "Find news about artificial intelligence"
        if st.button("⚖️ Regulatory & legal news"):
            st.session_state["prefill"] = "Find news about regulatory actions or legal issues"

        st.divider()
        st.markdown(f"**Model:** `{model_id}`")
        if st.button("🗑️ Clear chat"):
            st.session_state["messages"] = []
            st.rerun()

    # Load DB
    try:
        collection = load_collection()
        doc_count = collection.count()
        st.sidebar.success(f"Database loaded: {doc_count} articles")
    except Exception as e:
        st.error(
            f"Could not load ChromaDB. Did you run `build_db.py` first?\n\nError: {e}"
        )
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle prefill from sidebar buttons
    prefill = st.session_state.pop("prefill", None)

    # Chat input
    user_input = st.chat_input("Ask about the news…") or prefill

    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # Run agent
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking with {selected_model}…"):
                response_text, elapsed = run_agent(
                    user_input,
                    st.session_state["messages"][:-1],  # history excluding current
                    model_id,
                    collection,
                )
            st.markdown(response_text)
            st.caption(f"⏱️ {elapsed:.1f}s · {selected_model}")

        st.session_state["messages"].append(
            {"role": "assistant", "content": response_text}
        )


if __name__ == "__main__":
    main()