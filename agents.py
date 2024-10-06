import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from crewai import Agent
from tools import Tools

load_dotenv()

class Agents:
    def __init__(self):
        self.openaigpt4o = ChatOpenAI (
            model=os.getenv("OPENAI_MODEL_NAME"),
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
    def master_historian_agent(self):
        try:
            history = Agent(
                role="Master World Historian",
                goal="""Provide the latest and most specific information about world history.
                        Your task is to uncover detailed insights about historical figures,
                        major events, and their global impact.""",
                memo=False,
                backstory="""You are an expert in global history, with deep knowledge of key figures 
                            and significant events that shaped civilizations. Your mastery allows 
                            you to search for detailed historical accounts and provide specific 
                            insights into world events.""",
                tools=[Tools().search_tool()],
                verbose=True,
                llm=self.openaigpt4o
            )
            return history
        except Exception as e:
            print(f"Error creating master_historian_agent: {str(e)}")
        return None
    
    def researcher_historian_agent(self):
        try:
            research = Agent(
            role="World Historian Researcher",
            goal="""Gather the latest weather information from the location of the historical
                    event in question. Your job is to retrieve real-time data on the weather
                    from the specified historical sites.""",
            memo=False,
            backstory="""With your focus on historical contexts and environments, you are a researcher
                        who specializes in linking weather patterns with historical events. You use 
                        the latest weather APIs to ensure historical accuracy.""",
            tools=[Tools().weather_tool()],
            verbose=True,
            llm=self.openaigpt4o
        )
            return research
        except Exception as e:
            print(f"Error creating researcher_historian_agent: {str(e)}")
            return None
    
    def reporter_historian_agent(self):
        try:
            report = Agent(
            role="World Historian Reporter",
            goal="""Report on the latest news from the historical site in question.
                    Your job is to find the most relevant and up-to-date news related to
                    the specific historical locations.""",
            memo=False,
            backstory="""As a history-focused reporter, you are skilled at connecting current 
                        news with historical relevance. Your investigative skills allow you to 
                        find up-to-date stories from historical locations and report them accurately.""",
            tools=[Tools().news_finder()],
            llm=self.openaigpt4o                
            )
            return report
        except Exception as e:
            print(f"Error creating reporter_historian_agent: {str(e)}")
            return None
    
    def photographer_historian_agent(self):
        try:
            photo = Agent(
                role="World Historian Photographer",
                goal="""Obtain the latest images from the historical sites specified.
                        Your task is to capture the latest visual insights from these significant
                        locations.""",
                memo=False,
                backstory=""" As a visual storyteller, you are passionate about preserving the imagery 
                            of historical landmarks. With expertise in photographic documentation,
                            you use the latest image search technology to capture stunning visuals 
                            from historically significant sites.""",
                tools=[Tools().image_search()],
                llm=self.openaigpt4o
            )
            return photo
        except Exception as e:
            print(f"Error creating photographer_historian_agent: {str(e)}")
            return None
            
    def question_validator_agent(self):
        try:
            vquestion = Agent(
                role='Question Validator',
                goal='Validate if the user question is related to history.',
                verbose=False,
                memory=False,
                backstory=(
                    "As a historian and researcher, you are tasked with ensuring that "
                    "the user question is relevant to historical topics."
                ),
                tools=[Tools().search_tool()],
                llm=self.openaigpt4o
            )
            return vquestion
        except Exception as e:
            print(f"Error creating question_validator_agent: {str(e)}")
            return None

    def location_validator_agent(self):
        try:
            vlocation = Agent(
                role='Location Validator',
                goal='Validate if the input location is a real location on Earth.',
                verbose=False,
                memory=False,
                backstory=(
                    "As a geographer and cartographer, your expertise lies in identifying real "
                    "locations on Earth to ensure accurate location recognition."
                ),
                tools=[Tools().search_tool()],
                llm=self.openaigpt4o
            )
            return vlocation
        except Exception as e:
            print(f"Error creating location_validator_agent: {str(e)}")
            return None
    
    def language_validator_agent(self):
        try:
            vlanguage = Agent(
                role='Language Validator',
                goal='Validate if the input language is a real language.',
                verbose=False,
                memory=False,
                backstory=(
                    "As a linguist and language expert, your expertise lies in identifying real "
                    "languages to ensure accurate language recognition."
                ),
                tools=[Tools().search_tool()],
                llm=self.openaigpt4o
            )
            return vlanguage
        except Exception as e:
            print(f"Error creating language_validator_agent: {str(e)}")
            return None
