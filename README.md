CrewAI Jothidam RAG Assistant

A bilingual (English and Tamil) Jothidam question-answering application built using CrewAI, RAG, OpenAI, and Streamlit.

## Features

- Knowledge Agent: Retrieves answers from local Jothidam PDF documents.
- Web Research Agent: Independently searches the web and answers the same question.
- Record-Keeping Agent: Saves the question and both answers to answers_log.txt.
- Bilingual Support: Accepts questions in English and Tamil.
- Interactive UI: Displays both answers using Streamlit.

## Technologies

- Python
- CrewAI
- OpenAI API
- Serper API
- Streamlit
- FAISS
- Sentence Transformers

## Run the Application

bash
streamlit run app.py


## Example Questions

- What is Jothidam?
- What are the nine planets in Jothidam?
- What is a Rasi?
- What is a Nakshatra?
- ஜோதிடம் என்றால் என்ன?
- ராசி என்றால் என்ன?

## Logging

Each successful interaction is recorded in answers_log.txt, including the user's question and both agents' answers.
