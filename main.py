import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from crewai import Process, Crew
from agents import Agents
from tasks import Tasks

load_dotenv()

def main():
    st.set_page_config(
        page_title="GEMA AI | HistorianGPT",
        layout="centered",
        page_icon="🗺️",
    )
    
    # HTML/CSS Styling
    st.markdown("""
        <style>
            body {
                font-family: 'Arial', sans-serif;
                
            }
            .title {
                color: #001f3f;
                text-align: center;
                font-size: 3em;
                font-weight: bold;
                margin-bottom: 20px;
            }
            .subtitle {
                color: #007bff;
                text-align: center;
                margin-bottom: 40px;
            }
            .footer {
                text-align: center;
                font-size: 0.9em;
                color: #666;
                margin-top: 40px;
            }
            .stButton>button {
                background-color: #001f3f;
                color: white;
                font-size: 1.2em;
                padding: 10px 20px;
            }
            @media only screen and (max-width: 768px) {
                .title {
                    font-size: 2.5em;
                }
            }
        </style>
    """, unsafe_allow_html=True)

    st.image('gema.png', width=120)
    st.markdown('<h1 class="title">Welcome to HistorianGPT</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your personal AI historian at your service.</p>', unsafe_allow_html=True)

    with st.form("historian_form"):
        question = st.text_input("Question", placeholder="What historical event do you want to know about?", key="question", help="Ask any historical question.")
        location = st.text_input("Location", placeholder="Enter a specific location if relevant", key="location", help="Optional location for narrowing down the historical context.")
        language = st.text_input("Language", placeholder="Enter a language for the output", key="language", help="Choose the language for the answer.")
        submit_button = st.form_submit_button(label="Ask HistorianGPT")    
    
    if submit_button:
        api_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("OPENAI_MODEL_NAME")

        if not api_key or not model_name:
            st.error("API Key or Model Name missing from environment!")
            return

        openaigpt4 = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            api_key=api_key
        )
        
        inputs = {'question': question, 'location': location, 'language': language}
        
        try:
            validation_crew = Crew(
                agents=[Agents().question_validator_agent(), Agents().location_validator_agent(), Agents().language_validator_agent()],
                tasks=[Tasks().question_validate(), Tasks().location_validate(), Tasks().language_validate()],
                process=Process.sequential,
                manager_llm=openaigpt4
            )
            
            validation_result = validation_crew.kickoff(inputs=inputs)
            
            if all("yes" in result.lower() for result in validation_result):
                st.success("Validation successful! Proceeding with research...")
      
                researcher_crew = Crew(
                    agents=[Agents().master_historian_agent(), Agents().researcher_historian_agent(), Agents().reporter_historian_agent(), Agents().photographer_historian_agent()],
                    tasks=[Tasks().historical_task(), Tasks().weather_task(), Tasks().news_task(), Tasks().photo_task()],
                    process=Process.sequential,
                    manager_llm=openaigpt4
                )
                
                research_result = researcher_crew.kickoff(inputs=inputs)
                
                for result in research_result:
                    st.write(result)
            
            else:
                st.error("Validation failed. Please check your inputs.")
        
        except Exception as e:
            st.error(f"Error during crew execution: {str(e)}")
    else:
        st.info("Fill out the form to ask a question.")
