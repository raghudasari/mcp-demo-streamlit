import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient
from langchain.memory import ConversationBufferWindowMemory
from langsmith.wrappers import wrap_openai
import streamlit as st

load_dotenv()
CONFIG = {
    "mcpServers": {
        "community-search-tool": {
            "url": st.secrets["MCP_COMMUNITY_API_URL"],
            "headers":{
              "x-api-key": st.secrets["MCP_COMMUNITY_API_KEY"],
              "Authorization": f"Bearer {st.secrets["MCP_COMMUNITY_BEARER_TOKEN"]}"
            }
        }
    }
}



# Load OpenAI API key from Colab secrets and set environment variable
#openai_api_key = os.environ['OPENAI_API_KEY']
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
#langsmith_api_key = os.environ['LANGSMITH_API_KEY']
#os.environ['LANGSMITH_API_KEY'] = langsmith_api_key

# Setup Streamlit page
st.set_page_config(page_title="MCP Chatbot", layout="wide")
st.title("🏡 Lennar Community Assistant")

# Initialize session state for agent and messages
if "agent" not in st.session_state:
    client = MCPClient.from_dict(CONFIG)
    memory = ConversationBufferWindowMemory(k=3, return_messages=True)

    llm = ChatOpenAI(temperature=0)
    st.session_state.client = client
    st.session_state.agent = MCPAgent(llm=llm, client=client, max_steps=20)
    st.session_state.chat_history = []

# User input box
user_input = st.chat_input("Ask me about communities in Miami...")

if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)

    # Append to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Get response from MCP agent
    async def get_response(query):
        return await st.session_state.agent.run(query)

    response = asyncio.run(get_response(user_input))

    # Show bot response
    st.chat_message("assistant").markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Clean up sessions when app stops
def cleanup():
    try:
        asyncio.run(st.session_state.client.close_all_sessions())
    except:
        pass

# Optional: Manual reset button
if st.button("🔄 Reset Chat & Close Sessions"):
    try:
        asyncio.get_event_loop().run_until_complete(st.session_state.client.close_all_sessions())
    except:
        pass
    st.session_state.clear()
    st.experimental_rerun()
