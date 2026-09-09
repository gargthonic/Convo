import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.constants import START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from psycopg import Connection
from psycopg.rows import dict_row

from tools import tools

load_dotenv()
DB_URI = os.getenv(
    "DB_URI",
    "postgresql://postgres:password@localhost:5432/db_convo_chatbot_checkpoints?sslmode=disable",
)
DEFAULT_MODEL = os.getenv("CURRENT_MODEL", "poolside/laguna-s-2.1:free")

ALLOWED_MODELS = [
    "poolside/laguna-s-2.1:free",
    "inclusionai/ling-3.0-flash-fin:free",
    "thinkingmachines/inkling:free",
    "cohere/north-mini-code:free"
]

SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named Convo similar to ChatGPT.

You can:
1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool.
4. Search the web for latest/current information using Tavily Search.
5. Remember important user information using the memory tool.
6. Recall memory when useful.
7. Use calculator for math.

Rules:
- If the user asks about latest news, current events, recent updates, today's information, current prices, current people, current versions, new releases, or anything time-sensitive, use Tavily Search.
- If the user asks about an uploaded document, use search_uploaded_documents.
- If the user asks you to remember something, use remember_this.
- If the user asks about previous preferences or saved facts, use recall_memory.
- Use calculator for math questions.
- When using web search, summarize clearly and mention that the answer is based on web search results.
- Be clear, helpful, and concise.
"""


def normalize_model_name(model_name: str | None) -> str:
    """
    Validate selected model from frontend.
    If model is missing or not allowed, fallback to DEFAULT_MODEL.
    """

    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name


def build_agent(model_name: str) -> ChatOpenAI:
    """
    Build an agent based on selected model.
    """

    selected_model = normalize_model_name(model_name)

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=selected_model,
        timeout=30,
        max_retries=2,
        temperature=0.3,
        streaming=True
    )

    llm_with_tools = llm.bind_tools(tools)

    def chatbot_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

        response = llm_with_tools.invoke(messages)

        return {"messages": [response]}

    tool_node = ToolNode(tools=tools)

    graph = StateGraph(MessagesState)

    graph.add_node("chatbot", chatbot_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chatbot")
    graph.add_conditional_edges("chatbot", tools_condition)
    graph.add_edge("tools", "chatbot")

    conn = Connection.connect(DB_URI, autocommit=True, row_factory=dict_row)
    checkpointer = PostgresSaver(conn)
    # Required once — creates the checkpoint tables
    checkpointer.setup()
    agent = graph.compile(checkpointer=checkpointer)

    return agent


_AGENT_CACHE = {}


def get_agent(model_name: str | None = None) -> ChatOpenAI:
    """
    Return cached LangGraph agent for selected model.
    If not created yet, create it once and reuse it.
    """

    selected_model = normalize_model_name(model_name)

    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model] = build_agent(selected_model)

    return _AGENT_CACHE[selected_model]
