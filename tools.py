import os
from langchain.agents import Tool
from newsapi import NewsApiClient
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper, OpenWeatherMapAPIWrapper

load_dotenv()

os.environ["OPENWEATHERMAP_API_KEY"] = os.getenv("WEATHER_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["NEWS_API_KEY"] = os.getenv("NEWS_API_KEY")


# search_tool = GoogleSerperAPIWrapper(api_key=os.getenv("SERPER_API_KEY"))

# image_search = GoogleSerperAPIWrapper(type="images")

# weather_tool = OpenWeatherMapAPIWrapper()

news_finder = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

def get_latest_news(query):
    articles = news_finder.get_everything(q=query)
    return articles

def image_searching(query):
    image_search_tool = GoogleSerperAPIWrapper(api_key=os.getenv("SERPER_API_KEY") ,type="images")
    results = image_search_tool.results(query)
    return print(results)

class Tools:
    def search_tool(self):
        search_tool = Tool(
            name="Search",
            func=GoogleSerperAPIWrapper(api_key=os.getenv("SERPER_API_KEY")).run,
            description="Useful for when you need to answer questions about current events. You should ask targeted questions."
        )
        return search_tool
    
    # def image_search(self):
    #     try:
    #         return Tool(
    #             name="Image Search",
    #             func=image_searching,
    #             description="Useful for when you need to find images of something. Input should be a search query."
    #         )
    #     except Exception as e:
    #         print(f"Error creating Image Search tool: {str(e)}")
    #         return None
    
    def weather_tool(self):
        weather_tool = Tool(
            name="Weather",
            func=OpenWeatherMapAPIWrapper().run,
            description="Useful for when you need to find the weather of a location. Input should be a location."
        )
        return weather_tool
    
    def news_finder(self):
        news_tool = Tool(
            name="News Finder",
            func=get_latest_news,
            description="Useful for when you need to find the latest news of a location. Input should be a location."
        )
        return news_tool