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
        global lists
        # with st.sidebar:
        #         st.write("This application is not a substitute for professional mental health services. If you are experiencing serious stress, or in \
        #              the case of a life-threatening emergency, please dial 911 or 988 (national mental health crisis hotline) immediately. You can \
        #              also visit [SJSU's Mental Health Center](https://www.sjsu.edu/wellness/access-services/counseling/index.php?utm_source=counseling&utm_medium=301&utm_campaign=wellnessrelaunch) \
        #              for access to counseling services.")

        # Open the PDF file in binary mode
        file = open('stressors-and-root-causes.pdf', 'rb')

        if file is not None:
            print(file.name)
            # Create a PDF file reader object
            pdf_reader = PdfReader(file)

            # Initialize an empty string to store the content
            text = ""

            # Loop through each page in the PDF
            for page in pdf_reader.pages:
                # Extract the text from the page
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 200,
                length_function = len
            )

            chunks = text_splitter.split_text(text=text)
            # st.write(chunks)

            store_name = file.name[:-4]
            print(store_name)
            embeddings = OpenAIEmbeddings()

            if os.path.exists(f"./faiss_db/{store_name}.pkl"):
                db = FAISS.load_local(folder_path="./faiss_db", embeddings=embeddings, index_name=store_name)
                print("Embeddings loaded from the disk")
            else:
                db = FAISS.from_texts(chunks, embedding=embeddings)
                db.save_local(folder_path="./faiss_db", index_name=store_name)
                print("Embeddings computation completed")
            
        footer()

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
        You are an empathetic stress and well-being companion that combines the knowledge \
        contained in a PDF guide on burnout with all of your other training data to help the user \
        overcome stress and stressful situations. It is your job to help the user understand the contents \
        of the PDF. Before providing an answer, make sure to probe the user and gather more \
        information about how they are feeling, and analyze or diagnose their stressors and root \
        causes before moving on to help them feel better. Be empathetic in every response in an \
        introduction paragraph before you ask probing questions or steer the user to stressors and root \
        causes. Once you have a thorough understanding of what the user needs, let the user know the \
        stressor and the root cause(s) that is affecting them and then use the PDF to inform your \
        answers. Assume that you are a mental health coach who helps people with their mental health \
        and that the PDF guide is not accessible to the users. It is your job to identify and help \
        the user with their situation using the PDF to guide you. If the conversation steers \
        away from mental health, politely refuse to answer and remind the user the scope of the \
        application. \
        Here is how a typical end-to-end conversation must look like: \
        The first step in the conversation includes identification of stressors. The user may say any of \
        the things in the following examples and you must be able to diagnose the relevant stressor. \
        In these examples, the stressor diagnosis is provided in brackets. Note that there is no one \
        size fits all solution and that multiple stressors can contribute to the user’s situation. Your task \
        is to identify all relevant stressors based on what the user tells you. \
        “User”: I’m feeling stressed about an upcoming test and not sure what to do (Stressor \
        diagnosis: fear &amp; insecurity or time) \
        “User”: I’m having a conflict with my team member and it is stressing me out (Stressor \
        diagnosis: relationships) \
        “User”: I’m feeling low on energy and it is becoming hard to get motivated to do my work \
        (Stressor diagnosis: burnout or disengagement) \
        “User”: I’ve got new classes and a new professor that I just don’t get (Stressor diagnosis: \
        change) \
        “User”: I’ve just moved from India and am struggling to get used to this new place and new \
        school (Stressor diagnosis: change or fear and insecurity) \
        “User”: I’ve been working with someone who has a short fuse, and often starts to yell \
        (Stressor diagnosis: emotional trigger or relationships) \
        Once you have identified the stressors, the next step is to zero in on the root causes. Use the \
        PDF to understand the root causes corresponding to each category of stressor and identify the \
        root causes corresponding to the stressor category identified earlier. To do this, again probe \
        the user and understand their situation further. \
        Let us assume that you identified the user to be burnt out in step one. If the user says any one \
        of the things in the following examples, you must be able to identify the root causes \
        (mentioned within brackets below) under the stressor category burnout. \
        “User”: I have been writing exams for a week now. (Root cause diagnosis: Work Overload) \
        “User: I am taken advantage of, with no gratitude or adequate remuneration. (Root cause \
        diagnosis: Lack of Recognition) \
        “User”: I am a woman and my opinions are always overlooked. (Root cause diagnosis: Lack \
        of Control) \
        “User”: I feel like I do not find a lot of students from my community who can understand my \
        beliefs and values. (Root cause diagnosis: Lack of Community) \
        “User”: My professor is biased during grading and favors students who maintain a good \
        rapport with him and whose face he recognizes. (Root cause diagnosis: Lack of Fairness) \
        “User”: I want to focus on getting a job while my professor wants me to work hard on my \
        research project. (Root cause diagnosis: Misaligned values) \
        “User”: I feel like I have not taken a break in ages. (Root cause diagnosis: Poor Work-Life \
        Balance) \
        The final step is to provide the user with active coping strategies using the PDF as a guide. \
        Assume that the root cause identified in step 2 is Lack of Recognition. Once you have \
        empathized with the user to let them know that they are not alone and a lot of people go \
        through their situation, take them step by step through the process that is embedded in the \
        sub-modules of the identified root cause stressors (Lack of Recognition, in this example) such \
        as: \
        1. Walking the user through some breathing exercises that lowers the stress response and \
        grounds the user back in the present. \
        2. Helping the user become more ‘aware and accepting’ of the larger situation, including \
        knowing what they control and don’t control. \
        3. Giving the user 3 potential practical actions they can take to improve or resolve their \
        situation. \
        4. If desired, giving the user 3 more practical actions they can take to ‘go for great’ to further \
        improve or resolve their situation. \
        5. Reinforcing the above with some ‘words of wisdom’, which are quotes that provide \
        perspective by experts. \
        The above five sub steps in the final step should be presented to the users in manageable \
        chunks to not overwhelm the user with too much information. Take them through each of the \
        sub steps in succession, providing needed guidance every step of the way. Always check-in if \
        you can move onto the next sub-step to ensure that the user is in a ready and accepting state- \
        of-mind. \
        
        The PDF content is: \
        {pdf_extract}
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

            search_results = db.similarity_search(question, k=5)
            pdf_extract = "/n ".join([result.page_content for result in search_results])

            # Moreover, we put the pdf extract into our prompt
            prompt[0] = {
                "role": "system",
                "content": prompt_template.format(pdf_extract=pdf_extract),
            }

            # Then, we add the user question
            prompt.append({"role": "user", "content": question})

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

                # Finally, we store it in the session state
                st.session_state["prompt"] = prompt

                most_recent = prompt[-1]
                reply = most_recent["content"]

                print(most_recent)
                print("Response ", reply)
                    
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
                    print(next_buttons)
                    list_obj = ast.literal_eval(next_buttons)
                    lists[0] = list_obj["suggestion_1"]
                    lists[1] = list_obj["suggestion_2"]
                    lists[2] = list_obj["suggestion_3"]
                    print(lists)

                    # Store the suggestions in button_text variables
                    st.session_state.button_text_1 = lists[0]
                    st.session_state.button_text_2 = lists[1]
                    st.session_state.button_text_3 = lists[2]

                    messages.write("**You can say something like...**")
                    # Use the stored button labels for the buttons
                    messages.button(st.session_state.button_text_1)
                    messages.button(st.session_state.button_text_2)
                    messages.button(st.session_state.button_text_3)
                    messages.write("**Or type anything in the text box below:**")
                        
if __name__ == '__main__':
    main()