import os
import uuid
import queue
import threading
import streamlit as st
from PIL import Image
from openai import OpenAI
import mlflow

# Load environment variables
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT")
MODEL_NAME = os.getenv("MODEL_NAME", "tinyllama")

# MLflow prompt registry configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.redhat-ods-applications.svc.cluster.local:8443")
MLFLOW_PROMPT_NAME = os.getenv("MLFLOW_PROMPT_NAME")
MLFLOW_PROMPT_VERSION = os.getenv("MLFLOW_PROMPT_VERSION")
MLFLOW_TRACKING_TOKEN = os.getenv("MLFLOW_TRACKING_TOKEN")
if not os.getenv("MLFLOW_TRACKING_AUTH"):
    os.environ["MLFLOW_TRACKING_AUTH"] = "kubernetes"
if not os.getenv("MLFLOW_TRACKING_INSECURE_TLS"):
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

if not os.getenv("MLFLOW_WORKSPACE"):
    NAMESPACE_PATH = "/run/secrets/kubernetes.io/serviceaccount/namespace"
    if os.path.exists(NAMESPACE_PATH):
        with open(NAMESPACE_PATH) as f:
            os.environ["MLFLOW_WORKSPACE"] = f.read().strip()

SA_TOKEN_PATH = "/run/secrets/kubernetes.io/serviceaccount/token"
if os.path.exists(SA_TOKEN_PATH):
    with open(SA_TOKEN_PATH) as f:
        os.environ["MLFLOW_TRACKING_TOKEN"] = f.read().strip()

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("summarization")
mlflow.openai.autolog()

client = OpenAI(
    base_url=LLM_ENDPOINT + "/v1",
    api_key="no-key-required",
)

@st.cache_data(ttl=60)
def get_system_prompt():
    version = MLFLOW_PROMPT_VERSION or "latest"
    if version == "latest":
        prompt = mlflow.genai.load_prompt(f"prompts:/{MLFLOW_PROMPT_NAME}@{version}")
    else:
        prompt = mlflow.genai.load_prompt(f"prompts:/{MLFLOW_PROMPT_NAME}/{version}")
    return prompt.template

SYSTEM_PROMPT = get_system_prompt()


@mlflow.trace
def chat_completion(messages: list[dict], session_id: str, q: queue.Queue) -> str:
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.session": session_id,
        },
        tags={
            "mlflow.trace.session": session_id,
        },
    )
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=2048,
        temperature=0.9,
        stream=True,
    )
    full_response = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content is not None:
            full_response += content
            q.put(content)
    q.put(None)
    return full_response



# Page setup
st.set_page_config(
    page_title="Canopy - Educational Assistant",
    page_icon="🌿",
    layout="wide",
)

# Sidebar navigation
logo_path = "logo.png"
logo = Image.open(logo_path)
st.sidebar.image(logo, width="stretch")
st.sidebar.title("Canopy 🌿")
feature = st.sidebar.radio(
    "What do you want to do:",
    ["Summarization", "Information Search (coming soon)", "Student Assistant (coming soon)", "Socratic Tutor (coming soon)"],
    index=0
)

# Main view depending on feature
st.markdown("""
    <style>
    * {opacity:100% !important;}
    [data-testid="stSidebarNav"] {display: none;}
    /* Force chat message text to be black */
    .stTextArea textarea {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #2e8b57;'>Canopy</h1>
        <p style='font-size: 1.2em;'>Your leafy smart companion for education ✨</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

if feature == "Summarization":
    st.header("🌱 Chat with Summarizer")
    st.markdown("Have a conversation with the AI. Ask questions, request summaries, or continue the discussion!")

    # Initialize chat history and flags in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "awaiting_response" not in st.session_state:
        st.session_state.awaiting_response = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                # Escape HTML and ensure black text
                safe_content = msg["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")
                safe_content = safe_content.replace("\n", "<br>")
                st.markdown(f"""
                <div style='background-color: #e3f2fd; padding: 12px 15px; border-radius: 15px; margin: 8px 0; max-width: 80%; margin-left: auto;'>
                    <strong style='color: #1565c0;'>You</strong><br>
                    <span style='color: #000000;'>{safe_content}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Escape HTML characters and ensure black text
                safe_content = msg["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")
                safe_content = safe_content.replace("\n", "<br>")
                st.markdown(f"""
                <div style='background-color: #f1f8e9; padding: 12px 15px; border-radius: 15px; margin: 8px 0; max-width: 80%;'>
                    <strong style='color: #558b2f;'>Assistant</strong><br>
                    <span style='color: #000000;'>{safe_content}</span>
                </div>
                """, unsafe_allow_html=True)

        # Handle response if awaiting
        if st.session_state.awaiting_response:
            try:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for msg in st.session_state.chat_history:
                    messages.append({"role": msg["role"], "content": msg["content"]})

                q = queue.Queue()
                threading.Thread(
                    target=chat_completion,
                    args=(messages, st.session_state.session_id, q),
                    daemon=True,
                ).start()

                def stream_chunks():
                    while True:
                        chunk = q.get()
                        if chunk is None:
                            break
                        yield chunk

                st.markdown("<strong style='color: #558b2f;'>Assistant</strong>", unsafe_allow_html=True)
                full_response = st.write_stream(stream_chunks())

                if full_response:
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                st.session_state.awaiting_response = False
                st.rerun()

            except Exception as e:
                import sys
                print(f"[ERROR] {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                st.error(f"Something went wrong: {e}")
                st.session_state.awaiting_response = False
                st.rerun()

    # Add some spacing
    st.markdown("---")

    # Input section at the bottom (always shown)
    MAX_TOKENS = 4096

    # Use session state for input clearing
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0

    user_input = st.text_area("Your message:", height=100, key=f"chat_input_{st.session_state.input_key}", placeholder="Type your message here...")

    # Calculate tokens based on entire conversation + current message
    total_conversation = "\n".join([msg["content"] for msg in st.session_state.chat_history])
    total_text = total_conversation + "\n" + user_input
    approx_token_count = len(total_text) // 4
    tokens_left = MAX_TOKENS - approx_token_count - 50
    color = "red" if tokens_left <= 0 else ("orange" if tokens_left < 100 else "green")
    st.markdown(f"<span style='color:{color}; font-size: 0.9em;'>Tokens left (conversation): {tokens_left}</span>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        clear_chat = st.button("🗑️ Clear Chat", key="clear_chat", use_container_width=True)
        if clear_chat:
            st.session_state.chat_history = []
            st.session_state.awaiting_response = False
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.input_key += 1
            st.rerun()

    with col2:
        send_button = st.button("Send 💬", key="send_message", type="primary", use_container_width=True)

    # Check if we need to start streaming (phase 2)
    if st.session_state.get("start_streaming", False):
        st.session_state.start_streaming = False
        st.session_state.awaiting_response = True
        st.rerun()

    if send_button and not st.session_state.get("awaiting_response", False):
        if not user_input.strip():
            st.warning("Please enter a message.")
        elif not LLM_ENDPOINT:
            st.error("LLM_ENDPOINT not configured in environment variables.")
        elif tokens_left <= 0:
            st.error("Your message is too long. Please shorten it to stay within the token limit.")
        else:
            # Phase 1: Add message and clear input
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.input_key += 1
            st.session_state.start_streaming = True
            # Rerun to show empty input
            st.rerun()
else:
    st.info("This feature is coming soon. Stay tuned!")
