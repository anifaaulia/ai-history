import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from crewai import Agent
from tools import search_tool, weather_tool, news_finder, image_search

load_dotenv()

class Agents:
    def __init__(self):
        self.openaigpt4o = ChatOpenAI (
            model=os.getenv("OPENAI_MODEL_NAME"),
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
    def master_historian_agent(self):
        historian = Agent(
            role="Master World Historian",
            goal="""Provide insightful and accurate information about global historical events, 
                cultures, and figures. Use your knowledge to analyze historical contexts, 
                identify significant trends, and clarify historical narratives.""",
            backstory="""You are a distinguished historian with extensive knowledge of global history. 
                Your expertise encompasses various eras, cultures, and regions, enabling you to offer 
                comprehensive insights into the past. Your passion for historical research and analysis drives 
                your ability to connect historical events to broader themes and patterns.""",
            tools=[search_tool],
            verbose=True,
            llm=self.openaigpt4o
        )
        return historian
    
    def researcher_historian_agent(self):
        researcher = Agent(
            role="Master World Historian",
            goal=""" """,
            backstory=""" """,
            tools=[weather_tool],
            verbose=True,
            llm=self.openaigpt4o
        )