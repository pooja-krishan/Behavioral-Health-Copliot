# Behavioral Health Copilot
A Situational AI Co-Pilot to ‘Nip Stressors in the Bud’ and ‘In the Moment of Need’

# Details
This repository contains the codebase for a Behavioral Health Copilot, designed to democratize mental health access through innovative use of Prompt Engineering and Retrieval-Augmented Generation (RAG). 
The application is built with Streamlit and connected to MongoDB for storing anonymized user interaction data.

# Features
- **Mental Health Assistance:** Provides empathetic and insightful responses to users by diagnosing stressors, identifying root causes, and offering actionable coping strategies.
- **Prompt Engineering:** Uses carefully designed prompts to ensure accurate, empathetic, and contextually appropriate responses from the LLM.
- **Retrieval-Augmented Generation (RAG):** Contextualizes responses with proprietary PDF data stored in a vector database for precise and informed guidance.
- **User Interaction Logging:** Collects anonymized user interaction data in MongoDB to measure app performance and track its effectiveness.
- **Benchmarking:** Includes a separate main.py file implementing a prompt-only approach for comparison and performance benchmarking.
- **Conversation Continuity:** Prompts users to continue the conversation with suggested response buttons, ensuring a seamless and engaging interaction experience.

# Demo Video
[![Demo Video](https://img.youtube.com/vi/ldx1hQk7T4Y/0.jpg)](https://youtu.be/ldx1hQk7T4Y)

# Working
- The application begins by displaying buttons and/or a text box for user input
- Diagnoses stressors and identifies root causes using the context provided by the user
- Retrieves relevant proprietary data using RAG to generate informed responses
- Suggests actionable steps and coping mechanisms tailored to the user’s situation
- Logs user interactions anonymously for further analysis and performance tracking
- Suggests follow-up conversational prompts to keep the user engaged and assist further

# Setup
- Clone the repository

``` bash
git clone <repository_url>
cd <repository_directory>
```
- Install the dependencies
``` bash
pip install -r requirements.txt  
```
- Create a .env file in the root directory and add your OpenAI API key:
``` text
OPENAI_API_KEY=<your_openai_api_key>  
```
- Start MongoDB (ensure it is running on localhost:27017):
``` bash
mongod  
```
- Run the application
``` bash
streamlit run app.py  
```

# Usage
Usage
- Interact with the chatbot to identify stressors and root causes
- Receive context-specific guidance based on proprietary data (not provided in the repository due to confidentiality agreements; use the burnout-guide.pdf file for testing purposes)
- Anonymized user interactions are logged in MongoDB for analysis

# Benchmarking
- To compare the performance of prompt engineering with and without RAG and only prompt engineering, run the benchmarking script
``` bash
streamlit run main.py  
```
