import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Khởi tạo các tool
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

st.title("LangChain - Chat with search")

# Sidebar cho API key
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key: ", type="password")

if not api_key:
    st.warning("Please enter your Groq API Key in the sidebar.")
    st.stop()

# Khởi tạo session state cho messages nếu chưa có
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Hiển thị lịch sử chat
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Nhận input từ user
if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Khởi tạo LLM và agent
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="Deepseek-R1-Distill-Llama-70b",
        streaming=True
    )
    tools = [search, arxiv, wiki]
    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    # Gọi agent với prompt mới nhất
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(prompt, callbacks=[st_cb])
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.write(response)