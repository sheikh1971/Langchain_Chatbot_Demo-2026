
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# Optional: Set up LangSmith for tracing
LANGCHAIN_API_KEY=os.getenv("LANGSMITH_API_KEY")
LANGCHAIN_TRACING_V2="true"


# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer in at least 1 word and at most 5 lines with styling properly in bullets if needed."),
        ("human", "{input}"),
    ]
)


# Streamlit app
st.title("Langchain KutuGpt Demo")
user_input = st.text_input("search what you want to know")

# Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9)
output_parser = StrOutputParser()


chain = prompt | llm | output_parser 

if user_input:
    response = chain.invoke({"input": user_input})
    st.write(response)