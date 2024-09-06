from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
import streamlit as st
from weather_agent import WeatherAgent
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


# def get_agent(model: str = "gpt-4o-mini", api_key: str = os.environ["OPENAI_API_KEY"], temperature: float = 0):
#     """ Helper method to create and return a langgraph agent

#     Args:
#         model (str, optional): model to use. Defaults to "gpt-4o-mini".
#         api_key (str, optional): api_key . Defaults to os.environ["OPENAI_API_KEY"].
#         temperature (float, optional): temperature of the model. Defaults to 0.

#     Returns:
#         agent: a langgraph agent
#     """
#     # Create model
#     model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)

#     # Prepare search_tool
#     tavily_search_tool = TavilySearchResults(max_results=2)
#     tools = [tavily_search_tool]

#     # Prepare memory
#     memory = MemorySaver()

#     # Use them together to build the graph node
#     weather_agent = create_react_agent(model, tools=tools, checkpointer=memory)
#     return weather_agent


# Setting up the streamlit page
st.set_page_config(page_title="Weather Assistant")
st.header("Ask anything about weather")

# First row
r1c1, r1c2 = st.columns(2)
with r1c1:
    st.markdown("Ask anything about the weather in your city")

with r1c2:
    st.write(
        "Contact with [Author](https://github.com/sjchin88) to build your AI Projects")

# Get User Input
st.markdown("## Enter your question")


def get_question():
    question = st.text_area(label="Text", label_visibility='collapsed',
                            placeholder="Your question...", key="draft_input")
    return question


question = get_question()

if len(question) > 700:
    st.write("Please enter a shorter text. The maximum length is 700 words.")
    st.stop()


def compile_stream(stream):
    """ compile the stream messages and return a str representation

    Args:
        stream (_type_): A series of langgraph.pregel.io.AddableValuesDict object

    Returns:
        _type_: _description_
    """
    compiled_msg = ""
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, AIMessage):
            compiled_msg += message.pretty_repr()
            compiled_msg += "\n"
        else:
            continue
            # compiled_msg += message.pretty_repr()
            # compiled_msg += "\n"
    return compiled_msg


if question:
    weather_agent = WeatherAgent.get_agent()
    inputs = {"messages": [("user", f"{question}")]}
    # We need a config to store the session id
    config = {"configurable": {"thread_id": "1"}}
    output = weather_agent.stream(inputs, config=config, stream_mode="values")
    output_str = compile_stream(output)
    st.write(output_str)
