import streamlit as st
import asyncio
import os
import sys
import time
import threading
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

# Page Setup & Configuration
st.set_page_config(
    page_title="AI Agent - Math & PDF RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

script_dir = Path(__file__).resolve().parent
dotenv_path = script_dir / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Bridge Streamlit Cloud Secrets into os.environ
try:
    if hasattr(st, "secrets"):
        for key, value in st.secrets.items():
            if isinstance(value, str) and key not in os.environ:
                os.environ[key] = value
except Exception:
    pass

MCP_SERVER_URL = "http://127.0.0.1:8050/mcp"

# ---------------------------------------------------------
# Custom Styling (Glassmorphism & Modern UI)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Styles & Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* App Header Styling */
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6EE7B7 0%, #3B82F6 50%, #9333EA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .hero-subtitle {
        color: #9CA3AF;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    /* Sidebar Glassmorphism Card */
    .sidebar-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    
    /* Server Status Badge */
    .status-badge-online {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .status-badge-offline {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Tool Call Execution Card */
    .tool-execution-card {
        background-color: #0f172a;
        border-left: 4px solid #3b82f6;
        padding: 8px 12px;
        border-radius: 6px;
        margin: 4px 0;
        font-family: monospace;
        font-size: 0.88rem;
    }
    .tool-name { color: #60a5fa; font-weight: bold; }
    .tool-arg { color: #f59e0b; }
    .tool-res { color: #34d399; }
</style>
""", unsafe_allow_html=True)


def format_observation(obs):
    """Format tool output safely for audit logging."""
    if isinstance(obs, str):
        return obs if len(obs) <= 300 else obs[:300] + "..."
    return str(obs)[:300]

def check_mcp_server_online():
    """Ping FastMCP server HTTP endpoint."""
    try:
        req = urllib.request.Request(MCP_SERVER_URL, method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return True
    except urllib.error.HTTPError:
        # FastMCP streamable-http returns 406 or 405 on raw GET, proving it is alive and responding
        return True
    except Exception:
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', 8050))
            sock.close()
            return result == 0
        except Exception:
            return False

@st.cache_resource
def start_background_mcp_server():
    """Ensure FastMCP server is running in the background."""
    if not check_mcp_server_online():
        # First approach: run in a daemon thread directly inside this Python process
        try:
            from mcp_server import mcp
            thread = threading.Thread(
                target=mcp.run,
                kwargs={"transport": "streamable-http", "host": "0.0.0.0", "port": 8050},
                daemon=True
            )
            thread.start()
            for _ in range(25):
                time.sleep(0.3)
                if check_mcp_server_online():
                    break
            return thread
        except Exception:
            # Fallback approach: subprocess with explicit environment inheritance
            try:
                server_script = Path(__file__).resolve().parent / "mcp_server.py"
                process = subprocess.Popen(
                    [sys.executable, str(server_script)],
                    env=dict(os.environ)
                )
                for _ in range(25):
                    time.sleep(0.3)
                    if check_mcp_server_online():
                        break
                return process
            except Exception:
                return None
    return None

# Auto-launch FastMCP server on startup
start_background_mcp_server()

async def fetch_mcp_tools():
    """Connect to MCP server and fetch tools list."""
    client = MultiServerMCPClient(
        {
            "mcp_server_http": {
                'transport': 'streamable-http',
                'url': MCP_SERVER_URL,
            }
        }
    )
    return await client.get_tools()

def init_agent_executor(tools, api_key: str):
    """Initialize LangChain AgentExecutor with Groq model and MCP tools."""
    model = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        api_key=api_key
    )
    
    system_prompt = (
        """
        You are a highly reliable AI Assistant equipped with specific tools for math calculations, weather forecasting, and PDF document RAG.

        MATH CALCULATION RULES:
        1. You have ABSOLUTELY ZERO mental math capability. Every mathematical operation MUST be routed to tool calls.
        2. For ANY mathematical expression (e.g. '((76+45)*6)^2', '(90 + 456) / 3 * 2', '5!'), call `calculate_expression` FIRST with the mathematical expression string. It calculates expressions accurately in 1 step!
        3. ALWAYS pass actual numeric strings or valid mathematical formulas as tool arguments. NEVER pass text placeholders, words, or variable names like 'result', 'ans', 'output', 'previous_result', or 'x'.
        4. NEVER call PDF RAG or weather tools for math problems!

        STRICT DOCUMENT & PDF RAG RULES:
        1. For non-math questions, general concepts, topics, explanations, summaries, study notes, or questions, call PDF RAG tools (`search_pdf_knowledge`, `summarize_pdf_topic`, `generate_pdf_notes`, `generate_pdf_questions`) to query the indexed document.
        2. Base your final response strictly on the retrieved document chunks from the tool observation.

        STRICT DOMAIN ROUTING:
        1. Math Queries: Call ONLY `calculate_expression` or math tools (`addition`, `multiplication`, `subtraction`, `division`, `power`, `square_root`, `sine`, `cosine`, `tangent`, `factorial`, `degree_to_radian`, `get_constantvalue`). NEVER call PDF or weather tools for math!
        2. PDF / General Concept Queries: Call ONLY PDF RAG tools (`search_pdf_knowledge`, `summarize_pdf_topic`, `generate_pdf_notes`, `generate_pdf_questions`).
        3. Weather Queries: Call ONLY `weather` when explicitly asked for weather in a city.
        4. DO NOT call any tool that is not required for the user's prompt.
        """
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    
    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=10,
        handle_parsing_errors=True
    )
    
    return agent_executor


def run_agent_query(user_prompt: str, chat_history: list):
    """Executes agent prompt with dynamic document context injection and automatic fallback."""
    indexed_doc = st.session_state.get("indexed_pdf", None)
    chroma_dir = Path("./chroma_db")
    
    # Check if prompt is a math expression
    lower_prompt = user_prompt.lower()
    is_math = any(op in user_prompt for op in ['+', '*', '/', '^', '=']) or any(kw in lower_prompt for kw in ['calculate', 'compute', 'solve', 'factorial', 'sin', 'cos', 'tan', 'sqrt', 'power'])
    
    formatted_input = user_prompt
    if not is_math and (indexed_doc or (chroma_dir.exists() and any(chroma_dir.iterdir()))):
        doc_label = indexed_doc if indexed_doc else "Indexed PDF Document"
        formatted_input = (
            f"[ACTIVE DOCUMENT STATUS: '{doc_label}' is indexed in ChromaDB. "
            f"Use 'search_pdf_knowledge' to query document content for this question.]\n\n"
            f"User Question: {user_prompt}"
        )

    recent_history = chat_history[-4:] if chat_history else []

    try:
        response_obj = asyncio.run(
            st.session_state.agent_executor.ainvoke({
                "input": formatted_input,
                "chat_history": recent_history
            })
        )
        return response_obj.get("output", ""), response_obj.get("intermediate_steps", [])
    except Exception as e:
        err_str = str(e)
        active_key = os.getenv("GROQ_API_KEY") or st.session_state.get("user_groq_key", "")
        if "429" in err_str or "Rate limit" in err_str or "rate_limit_exceeded" in err_str:
            fallback_model = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.1,
                api_key=active_key
            )
            fallback_res = fallback_model.invoke(formatted_input)
            return fallback_res.content, []
        elif "Failed to call a function" in err_str or "failed_generation" in err_str or "failed" in err_str.lower():
            fallback_model = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.1,
                api_key=active_key
            )
            fallback_res = fallback_model.invoke(formatted_input)
            return fallback_res.content, []
        else:
            raise e




# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "langchain_chat_history" not in st.session_state:
    st.session_state.langchain_chat_history = []

if "indexed_pdf" not in st.session_state:
    st.session_state.indexed_pdf = None

# 1. Discover tools from FastMCP server independently
if not st.session_state.get("mcp_tools") and check_mcp_server_online():
    try:
        tools_list = asyncio.run(fetch_mcp_tools())
        st.session_state.mcp_tools = tools_list
        st.session_state.pop("tools_error", None)
    except Exception as e:
        st.session_state.mcp_tools = []
        st.session_state.tools_error = str(e)

# 2. Get active Groq API key from environment, secrets, or user input
active_groq_key = os.getenv("GROQ_API_KEY") or st.session_state.get("user_groq_key", "")

# 3. Initialize AgentExecutor if tools are loaded and key is provided
if st.session_state.get("agent_executor") is None and st.session_state.get("mcp_tools"):
    if active_groq_key:
        try:
            agent_exec = init_agent_executor(st.session_state.mcp_tools, active_groq_key)
            st.session_state.agent_executor = agent_exec
            st.session_state.pop("init_error", None)
        except Exception as e:
            st.session_state.agent_executor = None
            st.session_state.init_error = str(e)
    else:
        st.session_state.init_error = "GROQ_API_KEY is missing. Please enter your Groq API key in the sidebar below or add it to Streamlit Secrets."

# ---------------------------------------------------------
# Sidebar Layout
# ---------------------------------------------------------
with st.sidebar:
    st.title("Control Panel")
    
    # FastMCP Server Status Check
    server_online = check_mcp_server_online()
    st.subheader("Server Status")
    if server_online:
        tools_count = len(st.session_state.get("mcp_tools", []))
        st.markdown(f'<div class="status-badge-online">● FastMCP Server Online ({tools_count} tools)</div>', unsafe_allow_html=True)
        st.caption(f"Endpoint: `{MCP_SERVER_URL}`")
    else:
        st.markdown('<div class="status-badge-offline">● FastMCP Server Offline</div>', unsafe_allow_html=True)
        if "init_error" in st.session_state:
            with st.expander("Show Connection Details"):
                st.caption(st.session_state.init_error)
        if st.button("🚀 Start / Reconnect Server", type="primary", use_container_width=True):
            start_background_mcp_server.clear()
            start_background_mcp_server()
            st.session_state.pop("mcp_tools", None)
            st.session_state.pop("agent_executor", None)
            st.rerun()

    # Groq API Key Configuration
    st.subheader("🔑 Groq API Key")
    if os.getenv("GROQ_API_KEY") and not st.session_state.get("user_groq_key"):
        st.caption("✅ Loaded from environment / Streamlit Secrets")
    else:
        user_key = st.text_input(
            "Enter GROQ_API_KEY:",
            value=st.session_state.get("user_groq_key", ""),
            type="password",
            placeholder="gsk_...",
            help="Get a free API key at https://console.groq.com"
        )
        if user_key and user_key != st.session_state.get("user_groq_key"):
            st.session_state.user_groq_key = user_key
            os.environ["GROQ_API_KEY"] = user_key
            st.session_state.pop("agent_executor", None)
            st.rerun()

    if "init_error" in st.session_state and st.session_state.get("agent_executor") is None:
        with st.expander("⚠️ Agent Status", expanded=True):
            st.warning(st.session_state.init_error)
            if st.button("🔄 Initialize Agent", use_container_width=True):
                st.session_state.pop("agent_executor", None)
                st.rerun()

    st.divider()

    # RAG PDF Document Section
    st.subheader("📄 PDF Document Reader (RAG)")
    uploaded_pdf = st.file_uploader("Upload PDF File", type=["pdf"])
    if uploaded_pdf is not None:
        uploads_dir = Path("./temp_uploads")
        uploads_dir.mkdir(exist_ok=True)
        saved_pdf_path = uploads_dir / uploaded_pdf.name
        with open(saved_pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        
        if st.button("⚡ Index PDF in ChromaDB", use_container_width=True):
            if not server_online or st.session_state.agent_executor is None:
                st.error("MCP Server must be online to index documents.")
            else:
                with st.spinner(f"Chunking and embedding '{uploaded_pdf.name}'..."):
                    try:
                        abs_path = str(saved_pdf_path.resolve())
                        output_text, _ = run_agent_query(
                            f"Index the PDF document at absolute path '{abs_path}'",
                            []
                        )
                        st.session_state.indexed_pdf = uploaded_pdf.name
                        st.success(output_text or "PDF indexed successfully!")
                    except Exception as e:
                        st.error(f"Failed to index PDF: {e}")
    
    if st.session_state.indexed_pdf:
        st.info(f"📌 Currently Indexed: **{st.session_state.indexed_pdf}**")

    st.divider()

    # Registered MCP Tools View
    st.subheader("🛠️ Active MCP Tools")
    if st.session_state.get("mcp_tools"):
        st.caption(f"Loaded **{len(st.session_state.mcp_tools)}** tools via FastMCP")
        for tool in st.session_state.mcp_tools:
            with st.expander(f"⚙️ `{tool.name}`"):
                st.write(tool.description or "No description provided.")
    else:
        if "tools_error" in st.session_state:
            st.error(f"Error discovering tools: {st.session_state.tools_error}")
        else:
            st.info("No tools currently loaded. Click 'Fetch Tools Now' below.")
        if st.button("🔄 Fetch Tools Now", use_container_width=True):
            st.session_state.pop("mcp_tools", None)
            st.rerun()

    st.divider()
    
    # Clear Session History
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.langchain_chat_history = []
        st.rerun()

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.markdown('<div class="hero-title">AI Agent - Math & PDF RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Autonomous AI Agent for Scientific Math & Document Retrieval-Augmented Generation</div>', unsafe_allow_html=True)

tab_chat, tab_rag = st.tabs(["💬 Conversational Agent", "📚 RAG Topic Analysis"])

with tab_chat:
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "steps" in message and message["steps"]:
                with st.expander("🛠️ Tool Executions & Routing Audit Log", expanded=False):
                    for step in message["steps"]:
                        action, observation = step
                        tool_name = getattr(action, 'tool', 'Unknown Tool')
                        tool_input = getattr(action, 'tool_input', {})
                        tool_res = format_observation(observation)
                        st.markdown(
                            f"""
                            <div class="tool-execution-card">
                                ▶ Call: <span class="tool-name">{tool_name}</span>(<span class="tool-arg">{tool_input}</span>)<br>
                                ➔ Result: <span class="tool-res">{tool_res}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            st.markdown(message["content"])

    # Chat Input Box
    if user_prompt := st.chat_input("Ask a math calculation or a question about your uploaded PDF..."):
        if not server_online:
            st.error("Cannot process query: FastMCP server is currently offline. Please launch `mcp_server.py`.")
        elif st.session_state.agent_executor is None:
            st.error(f"Agent failed to initialize. Error: {st.session_state.get('init_error', 'Unknown Error')}")
        else:
            # Append User Message to UI State
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            # Generate Assistant Response
            with st.chat_message("assistant"):
                with st.spinner("Routing query to Agent & Tools..."):
                    try:
                        output_text, intermediate_steps = run_agent_query(
                            user_prompt,
                            st.session_state.langchain_chat_history
                        )
                        
                        # Update LangChain memory history
                        st.session_state.langchain_chat_history.append(HumanMessage(content=user_prompt))
                        st.session_state.langchain_chat_history.append(AIMessage(content=output_text))

                        # Render Intermediate Tool Steps if present
                        if intermediate_steps:
                            with st.expander("🛠️ Tool Executions & Routing Audit Log", expanded=True):
                                for step in intermediate_steps:
                                    action, observation = step
                                    tool_name = getattr(action, 'tool', 'Unknown Tool')
                                    tool_input = getattr(action, 'tool_input', {})
                                    tool_res = format_observation(observation)
                                    st.markdown(
                                        f"""
                                        <div class="tool-execution-card">
                                            ▶ Call: <span class="tool-name">{tool_name}</span>(<span class="tool-arg">{tool_input}</span>)<br>
                                            ➔ Result: <span class="tool-res">{tool_res}</span>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                        
                        # Render Final Response Text
                        st.markdown(output_text)

                        # Store in UI Session State
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": output_text,
                            "steps": intermediate_steps
                        })

                    except Exception as e:
                        error_msg = f"An error occurred while executing the query: {str(e)}"
                        st.error(error_msg)

with tab_rag:
    st.subheader("Deep Topic Analysis & Study Tools")
    st.markdown("Perform dedicated analysis on your indexed PDF document for specific topics.")
    
    topic_query = st.text_input("Enter Topic Name", placeholder="e.g. Functions, Object Oriented Programming, Data Structures")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 Summarize Topic", use_container_width=True):
            if not topic_query.strip():
                st.warning("Please enter a topic name.")
            elif not server_online or st.session_state.agent_executor is None:
                st.error("MCP Server is offline.")
            else:
                with st.spinner(f"Generating summary for topic '{topic_query}'..."):
                    out, _ = run_agent_query(
                        f"Summarize the topic '{topic_query}' using the uploaded PDF document.",
                        []
                    )
                    st.markdown("### Topic Summary")
                    st.markdown(out)

    with col2:
        if st.button("📌 Create Short Notes", use_container_width=True):
            if not topic_query.strip():
                st.warning("Please enter a topic name.")
            elif not server_online or st.session_state.agent_executor is None:
                st.error("MCP Server is offline.")
            else:
                with st.spinner(f"Creating short bullet notes for topic '{topic_query}'..."):
                    out, _ = run_agent_query(
                        f"Generate short bullet-point study notes for the topic '{topic_query}' from the uploaded PDF document.",
                        []
                    )
                    st.markdown("### Bullet-Point Study Notes")
                    st.markdown(out)

    with col3:
        if st.button("❓ Generate Practice Questions", use_container_width=True):
            if not topic_query.strip():
                st.warning("Please enter a topic name.")
            elif not server_online or st.session_state.agent_executor is None:
                st.error("MCP Server is offline.")
            else:
                with st.spinner(f"Generating study questions for topic '{topic_query}'..."):
                    out, _ = run_agent_query(
                        f"Generate 3 to 5 open-ended subjective practice questions for the topic '{topic_query}' based on the uploaded PDF document.",
                        []
                    )
                    st.markdown("### Practice & Subjective Questions")
                    st.markdown(out)

