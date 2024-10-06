import os
import json
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
    
    st.markdown("""
        <style>
            body {
                font-family: 'Arial', sans-serif;
                
            }
            .title {
                color: #FFFFFF;
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
    st.markdown('<h1 class="title">GEMA AI | HistorianGPT</h1>', unsafe_allow_html=True)

    with st.form("historian_form"):
        question = st.text_input("Question", placeholder="What historical event do you want to know about?", key="question", help="Ask any historical question.")
        location = st.text_input("Location", placeholder="Enter a specific location if relevant", key="location", help="Optional location for narrowing down the historical context.")
        language = st.text_input("Language", placeholder="Enter a language for the output", key="language", help="Choose the language for the answer.")
        submit_button = st.form_submit_button(label="Ask HistorianGPT")    
        
    agents = Agents()
    tasks = Tasks()
    
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
                agents=[
                    agents.question_validator_agent(),
                    agents.location_validator_agent(),
                    agents.language_validator_agent()
                ],
                tasks=[
                    tasks.question_validate(question),  
                    tasks.location_validate(location),  
                    tasks.language_validate(language)
                ],
                process=Process.sequential,
                manager_llm=openaigpt4
            )

            
            validation_result = validation_crew.kickoff(inputs=inputs)
            
            if str(validation_result).lower().strip() == "yes":
                st.success("Validation successful! Proceeding with research...")
      
                researcher_crew = Crew(
                    agents=[
                            agents.master_historian_agent(),
                            agents.reporter_historian_agent()
                        ],
                    tasks=[
                            tasks.historical_task(question, location, language),
                            tasks.news_task(question, location, language),
                            tasks.summarize(question, location, language)
                        ],                    
                    process=Process.sequential,
                    manager_llm=openaigpt4
                )
                
                crew_output = researcher_crew.kickoff()
                                
                st.markdown(f"""
                            {crew_output}
                            """)
            
            else:
                st.error("Validation failed. Please check your inputs.")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
    else:
        st.info("Fill out the form to ask a question.")

if __name__ == "__main__":
    main()