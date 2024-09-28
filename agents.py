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
        history = Agent(
            role="Master World Historian",
            goal="""Provide the latest and most specific information about world history.
                    Your task is to uncover detailed insights about historical figures,
                    major events, and their global impact.""",
            backstory="""You are an expert in global history, with deep knowledge of key figures 
                        and significant events that shaped civilizations. Your mastery allows 
                        you to search for detailed historical accounts and provide specific 
                        insights into world events.""",
            tools=[search_tool],
            verbose=True,
            llm=self.openaigpt4o
        )
        return history
    
    def researcher_historian_agent(self):
        research = Agent(
            role="World Historian Researcher",
            goal="""Gather the latest weather information from the location of the historical
                    event in question. Your job is to retrieve real-time data on the weather
                    from the specified historical sites.""",
            backstory="""With your focus on historical contexts and environments, you are a researcher
                        who specializes in linking weather patterns with historical events. You use 
                        the latest weather APIs to ensure historical accuracy.""",
            tools=[weather_tool],
            verbose=True,
            llm=self.openaigpt4o
        )
        return research
    
    def reporter_historian_agent(self):
        report = Agent(
            role="World Historian Reporter",
            goal="""Report on the latest news from the historical site in question.
                    Your job is to find the most relevant and up-to-date news related to
                    the specific historical locations.""",
            backstory="""As a history-focused reporter, you are skilled at connecting current 
                        news with historical relevance. Your investigative skills allow you to 
                        find up-to-date stories from historical locations and report them accurately.""",
            tools=[news_finder],
            llm=self.openaigpt4o                
        )
        return report
    
    def photographer_historian_agent(self):
        photo = Agent(
            role="World Historian Photographer",
            goal="""Obtain the latest images from the historical sites specified.
                    Your task is to capture the latest visual insights from these significant
                    locations.""",
            backstory=""" As a visual storyteller, you are passionate about preserving the imagery 
                        of historical landmarks. With expertise in photographic documentation,
                        you use the latest image search technology to capture stunning visuals 
                        from historically significant sites.""",
            tools=[image_search],
            llm=self.openaigpt4o
        )