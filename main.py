from dotenv import load_dotenv
import openai

from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from PyPDF2 import PdfReader

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

from openai import OpenAI

from pymongo import MongoClient
import uuid
from datetime import datetime

import ast

from htbuilder import div, hr, a, p, styles
from htbuilder.units import percent, px

import json

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Get API key from environment variable
openai.api_key = OPENAI_API_KEY

lists = ["I am very stressed and tired", "I am demotivated", "I don't know how I feel"]

def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      footer {visibility: hidden;}
     .stApp { bottom: 15vh; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(0,0,0,0),
        border_style="inset",
        border_width=px(2),
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "If you are experiencing serious stress, or in the case of a life-threatening emergency, seek immediate help: Dial 911, Dial 988, or visit ",
        link("https://www.sjsu.edu/wellness/access-services/counseling/index.php?utm_source=counseling&utm_medium=301&utm_campaign=wellnessrelaunch", "SJSU's Mental Health Center"),
        "."
    ]
    layout(*myargs)
    

def main():
    if "session" not in st.session_state:
        st.session_state.checkbox = False
        st.session_state.session = False

    if not st.session_state.session:
        block = st.empty()
        with block.container():
            st.write("This app is an **experimental project that uses Generative AI** to provide mental well-being \
                         coaching for students at SJSU who are facing stress and stressful situations. This application \
                         is **NOT a substitute for professional mental health services**. While it aims to assist students at \
                         SJSU in coping with stress and challenging situations, it is essential to recognize its limitations. \
                         The app serves as a **temporary solution—akin to a band-aid offering active coping strategies** to address \
                         stressors in the moment of need.")
            st.write("The app does **NOT diagnose, treat, or prevent any mental health condition**, and it **does NOT provide any \
                         medical advice**. The app is NOT intended to replace or interfere with your relationship with your mental \
                         health provider.")
            st.write("The app is provided 'as is' and 'as available' **WITHOUT any warranty of any kind**, either express or implied. \
                         The app may contain errors, inaccuracies, or omissions, and it may not be suitable for your specific situation. \
                         The app may also have technical issues or limitations that affect its functionality or availability. **You are solely \
                         responsible for your use of the app and any consequences that may arise from it. You agree to indemnify and hold \
                         harmless the app developers, SJSU, and any other parties involved in the creation or distribution of the app from \
                         any claims, damages, liabilities, or losses that may result from your use of the app.**")
            st.write("This app **collects and analyzes your activity and interaction data**, such as the buttons you click and the length of \
                         your input, to improve user experience and the quality of the app. **We DO NOT sell or share this data with any \
                         third parties for advertising purposes.** We only use this data for our own internal research and development. **By using \
                         this app, you consent to our user interaction data collection and use.**")
            st.write("**If** you find yourself experiencing **serious stress** or facing a **life-threatening emergency, please seek immediate help:**")
            st.markdown("1.	**Dial 911:** For urgent medical assistance.")
            st.markdown("2.	**Dial 988:** The national mental health crisis hotline.")
            st.markdown("3.	**Visit [SJSU's Mental Health Center:](https://www.sjsu.edu/wellness/access-services/counseling/index.php?utm_source=counseling&utm_medium=301&utm_campaign=wellnessrelaunch)** \
                            Access professional counseling services on campus.")
            st.write("Remember that this app does not replace the expertise of mental health professionals. Prioritize your well-being and \
                         seek professional assistance when necessary. Your mental health is important and you deserve to get the help you need. \
                         By using this app, you agree to all the terms listed above.")
            st.write("Please click the button below to acknowledge and digitally sign that you have read and understood \
                     this information.")
            st.session_state.checkbox = st.checkbox("I agree and have understood the above terms")
            if st.button("Submit"):
                if not st.session_state.checkbox:
                    st.warning('Please agree to the terms to proceed.')
                    st.stop()
                st.session_state.session = True
                block.empty()
    if st.session_state.checkbox and st.session_state.session:
        footer()

        global lists

        with st.container():
            # Create a placeholder container that holds all the messages:
            messages = st.container()
        text_box = st.chat_input("How can I help you?")
            # disclaimer = st.container(height=100)

        # disclaimer.write("This **AI app** provides stress coping strategies for SJSU students. It **does NOT replace professional mental health services** or diagnose conditions. \
        # The app is 'as is' **WITHOUT warranty**. You are responsible for your use and agree to indemnify all parties involved. \
        # The app **collects your interaction data** for internal use. **We DO NOT sell or share this data.** By using this app, you consent to our data collection. \
        # **In emergencies, seek immediate help:** Dial 911, Dial 988, or visit [SJSU's Mental Health Center](https://www.sjsu.edu/wellness/access-services/counseling/index.php?utm_source=counseling&utm_medium=301&utm_campaign=wellnessrelaunch) \
        # By using this app, you agree to all the terms listed above.")

        # Store the button labels in session_state if they don't exist
        if 'button_text_1' not in st.session_state:
            st.session_state.button_text_1 = lists[0]
        if 'button_text_2' not in st.session_state:
            st.session_state.button_text_2 = lists[1]
        if 'button_text_3' not in st.session_state:
            st.session_state.button_text_3 = lists[2]
            
        # Initialize databse connection.
        # Connect to MongoDB
        client = MongoClient('mongodb://localhost:27017/')
        database = client['campus_companion']

        if 'UUID' not in st.session_state:
            st.session_state.UUID = str(uuid.uuid4())
        suggestion_1, suggestion_2, suggestion_3, text_box_input = False, False, False, False

        # Use the stored button labels for the buttons
        if messages.button(st.session_state.button_text_1):
            st.session_state.question = st.session_state.button_text_1
            suggestion_1 = True
        elif messages.button(st.session_state.button_text_2):
            st.session_state.question = st.session_state.button_text_2
            suggestion_2 = True
        elif messages.button(st.session_state.button_text_3):
            st.session_state.question = st.session_state.button_text_3
            suggestion_3 = True
        elif text_box:
            st.session_state.question = text_box
            text_box_input = len(text_box)
        else:
            st.session_state.question = None

        question = st.session_state.question
        print("Question: ",question)

        prompt_template = """
        You are an empathetic stress and well-being companion that helps the user overcome \
        stress and stressful situations. Before providing an answer, make sure to probe the \
        user and gather more information about how they are feeling, and analyze or diagnose \
        their condition before moving on to help them feel better. Once you have a thorough \
        understanding of what the user needs, use your knowledge to help the user overcome \
        their situation and actively cope with it. Assume that you are a well-being coach \
        who helps people with their stress and stressful situations and that your knowledge is not accessible \
        to users. It is your job to identify and help the user with their situation. If the \
        conversation steers away from stress and well-being, politely refuse to answer and remind the \
        user about the scope of the application.
        """

        # When calling ChatGPT, we  need to send the entire chat history together
        # with the instructions. ChatGPT doesn't know anything about
        # your previous conversations so you need to supply that yourself.
        # Since Streamlit re-runs the whole script all the time we need to load and
        # store our past conversations in a session state.
        prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])
        print(prompt)

        # Here we display all messages so far in our convo
        for message in prompt:
            # If we have a message history, let's display it
            if message["role"] != "system":
                with messages.chat_message(message["role"]):
                    st.write(message["content"])
            
        # If the user asks a question
        if question is not None:
            # Enter user event into the database
            interaction_data = {
            'UUID': st.session_state.UUID,
            'suggestion_1': suggestion_1,
            'suggestion_2': suggestion_2,
            'suggestion_3': suggestion_3,
            'text_box_input' : text_box_input,
            'timestamp': datetime.utcnow()  # Store timestamp as BSON date type
            }
            database.user_info.insert_one(interaction_data)

            # Moreover, we put the pdf extract into our prompt
            prompt[0] = {
                "role": "system",
                "content": prompt_template,
            }

            # Then, we add the user question
            prompt.append({"role": "user", "content": question})
            
            # Open the file in append mode ('a')
            with open('answer.json', 'a') as file:
                # Convert the dictionary to a JSON string
                json_str = json.dumps({"role": "user", "content": question})
                
                # Write the JSON string to the file
                file.write(json_str + "\n")

            # And make sure to display the question to the user
            with messages.chat_message("user", avatar="🧑"):
                st.write(question)

            with messages.chat_message("assistant", avatar="🤖"):
                botmsg = st.empty()  # This enables us to stream the response as it comes

                # Here we call ChatGPT with streaming
                response = []
                result = ""
                Client = OpenAI()
                for chunk in Client.chat.completions.create(
                    model="gpt-3.5-turbo", messages=prompt, stream=True
                ):
                    text = chunk.choices[0].delta.content
                    if text is not None:
                        response.append(text)
                        result = "".join(response).strip()

                        # Let us update the Bot's answer with the new chunk
                        botmsg.write(result)

                # When we get an answer back we add that to the message history
                prompt.append({"role": "assistant", "content": result})
                # Open the file in append mode ('a')
                with open('answer.json', 'a') as file:
                    # Convert the dictionary to a JSON string
                    json_str = json.dumps({"role": "assistant", "content": result})
                    
                    # Write the JSON string to the file
                    file.write(json_str + "\n")
                # Finally, we store it in the session state
                st.session_state["prompt"] = prompt

                most_recent = prompt[-1]
                reply = most_recent["content"]

                print(most_recent)
                print("Response ", reply)

                print("PROMPT ", prompt)
                        
                if most_recent["role"] == "assistant":
                    # Pass in the response as input to the model and ask for suggestions 
                    #to prompt the user to continue the conversation
                    prompt_probe = """
                    You are a helpful assistant who gives the user examples of things to say to keep the \
                    conversation flowing with a chatbot mental well being coach. Given the text, suggest \
                    3 very short and relevant response examples (15 to 20 words only) the user can say to \
                    respond to the text while simultaneously mentioning their situation and seeking for \
                    help from the chatbot OR probing the chatbot to help them further depending on the text. \
                    Do not suggest compliments or gratitude messages. Do not ask the chatbot mental well being \
                    coach about its experience and instead suggest some examples the user can say to share their \
                    experience and get help. Return these three suggestions as the value of a JSON object with three \
                    keys - "suggestion_1", "suggestion_2", and "suggestion_3". The answer returned by you must be a \
                    JSON object with 3 key-value pairs. Since the user is conversing with a chatbot, steer \
                    away from suggestions asking about the chatbot's experiences since it is a machine with no feelings. 

                    The text is:
                    {text}
                    """
                    suggestion = Client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                            "role": "user",
                            "content": prompt_probe.format(text=reply)
                            }
                            ]
                        )
                    next_buttons = suggestion.choices[0].message.content
                    list_obj = ast.literal_eval(next_buttons)
                    lists[0] = list_obj["suggestion_1"]
                    lists[1] = list_obj["suggestion_2"]
                    lists[2] = list_obj["suggestion_3"]
                    print(lists)

                    # Store the suggestions in button_text variables
                    st.session_state.button_text_1 = lists[0]
                    st.session_state.button_text_2 = lists[1]
                    st.session_state.button_text_3 = lists[2]

                    # Use the stored button labels for the buttons
                    messages.write("**You can say something like...**")
                    # Use the stored button labels for the buttons
                    messages.button(st.session_state.button_text_1)
                    messages.button(st.session_state.button_text_2)
                    messages.button(st.session_state.button_text_3)
                    messages.write("**Or type anything in the text box below:**")

if __name__ == '__main__':
    main()