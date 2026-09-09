from langchain_core.messages import HumanMessage

from agent import get_agent, DEFAULT_MODEL

agent = get_agent(DEFAULT_MODEL)

config = {
    "configurable": {
        "thread_id": "test_thread_id"
    }
}

for message_chunk, metadata in agent.stream(
        {"messages": [
            HumanMessage(content="My name is Pranshu. Remeber that")]},
        config=config,
        stream_mode="messages"
):
    print(message_chunk.content, flush=True, end="")
