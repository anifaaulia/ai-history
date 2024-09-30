import os
from newsapi import NewsApiClient
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper, OpenWeatherMapAPIWrapper

load_dotenv()

os.environ["OPENWEATHERMAP_API_KEY"] = os.getenv("WEATHER_API_KEY")

search_tool = GoogleSerperAPIWrapper()

image_search = GoogleSerperAPIWrapper(type="images")

weather_tool = OpenWeatherMapAPIWrapper()

news_finder = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))