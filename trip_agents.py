from crewai import Agent, tools
import re 
import streamlit as st
from langchain_core.language_models.chat_models import BaseChatModel
from crewai import LLM

from tools.browser_tools import BrowserTool
from tools.calculator_tools import CalculatorTool
from tools.search_tools import SearchTool

class TripAgents():
    def __init__(self, llm: BaseChatModel = None):
        self.llm = llm
        
        # Set up the tools for the agents
        self.search_tool = SearchTool()
        self.browser_tool = BrowserTool()
        self.calculator_tool = CalculatorTool()
        
    def city_selection_agent(self):
        """
        Agent to select a city for the trip.
        """
        return Agent(
            role='City Selection Expert',
            goal="Select the best city for a trip based on weather, season and prices",
            backstory='An expert in analyzing weather, season and prices to select the best city for a trip.',
            tools=[self.search_tool, self.calculator_tool],
            allow_delegation=False,
            llm= self.llm,
            verbose=True
        )
        
    def local_expert(self):
        """
        Agent to provide local expert advice on the selected city.
        """
        return Agent(
            role='Local Expert at this city',
            goal="Provide local expert advice on the selected city.",
            backstory="""A knowledgeable local expert who can provide insights and recommendations about the city.""",
            tools=[self.search_tool, self.browser_tool],
            allow_delegation=False,
            llm= self.llm,
            verbose=True
        )
        
    def travel_coincerge(self):
        """
        Agent to provide travel concierge services.
        """
        return Agent(
            role='Travel Concierge',
            goal="Create the most amazing travel itineraries with budget and packing suggestions for thec city",
            backstory="""A travel concierge who can assist with travel arrangements, recommendations, and bookings.""",
            tools=[self.search_tool, self.browser_tool, self.calculator_tool],
            allow_delegation=False,
            llm= self.llm,
            verbose=True
        )