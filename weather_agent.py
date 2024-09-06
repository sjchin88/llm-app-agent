"""
Module to manage weather_agent
"""
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


class WeatherAgent(object):
    __instance = None

    @staticmethod
    def get_agent():
        if WeatherAgent.__instance is None:
            print('Creating new agent')
            WeatherAgent.__instance = WeatherAgent._create_agent()
        return WeatherAgent.__instance

    @staticmethod
    def _create_agent(model: str = "gpt-4o-mini", api_key: str = os.environ["OPENAI_API_KEY"], temperature: float = 0):
        """ Helper method to create and return a langgraph agent 

        Args:
            model (str, optional): model to use. Defaults to "gpt-4o-mini".
            api_key (str, optional): api_key . Defaults to os.environ["OPENAI_API_KEY"].
            temperature (float, optional): temperature of the model. Defaults to 0.

        Returns:
            agent: a langgraph agent
        """
        # Create model
        model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)

        # Prepare search_tool
        tavily_search_tool = TavilySearchResults(max_results=2)
        tools = [tavily_search_tool]

        # Prepare memory
        memory = MemorySaver()

        # Use them together to build the graph node
        weather_agent = create_react_agent(
            model, tools=tools, checkpointer=memory)
        return weather_agent
