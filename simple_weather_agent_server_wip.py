#!/usr/bin/env python
"""Example of a chat server with persistence (memory) handled on the backend.
Reference and modified based on https://github.com/langchain-ai/langserve/blob/main/examples/chat_with_persistence/server.py

For simplicity, we're using file storage here -- to avoid the need to set up
a database. This is obviously not a good idea for a production environment,
but will help us to demonstrate the RunnableWithMessageHistory interface.

We'll use cookies to identify the user and/or session. This will help illustrate how to
fetch configuration from the request.
"""
import re
from pathlib import Path
from typing import Callable, Union

from fastapi import FastAPI, HTTPException
from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.agent import AgentOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

import os

_ = load_dotenv(find_dotenv())

openai_api_key = os.environ["OPENAI_API_KEY"]

web_app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

# Declare the prompt, model required for a chain
system_template = "Translate the following into {language}:"

""" prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
]) """

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're an assistant by the name of Goku"),
        ("human", "{human_input}"),
    ]
)

# Prepare search_tool
tavily_search_tool = TavilySearchResults(max_results=2)
tools = [tavily_search_tool]

# The agent returned is a runnable
weather_agent = create_react_agent(model, tools)

# From https://cookbook.openai.com/examples/how_to_build_a_tool-using_agent_with_langchain


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:

        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split(
                    "Final Answer:")[-1].strip()},
                log=llm_output,
            )

        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        # If it can't parse the output it raises an error
        # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)

        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


output_parser = CustomOutputParser()

# So we can try to chain it
chain = prompt | weather_agent | output_parser

web_app = FastAPI(
    title="simple backend with agent",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

add_routes(
    web_app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(web_app, host="localhost", port=8000)
