import streamlit as st
from groq import Groq

# Connect to Groq API
chatbot = Groq(api_key="")

st.markdown("""
    <style>
        body, .stApp {
            background-color: #FFEFF5;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("MamaPulse - Diet & Nutrition Assistant")
st.write("Hello! I'm here to help with your diet, healthy recipes, and nutrition queries. Let’s talk food and fitness!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "system",
            "content": (
                "You're a diet and nutrition expert. Limit your responses strictly to healthy eating, food benefits, diets, "
                "and meal planning. If asked about anything else, gently guide the user back to food-related topics."
            )
        }
    ]

for msg in st.session_state.chat_history:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

prompt = st.chat_input("Got a food or diet question?")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    try:
        reply = chatbot.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=st.session_state.chat_history
        ).choices[0].message.content

        if all(word not in reply.lower() for word in ["diet", "nutrition", "meal", "food"]):
            reply = "⚠️ I'm your diet and nutrition buddy. Please keep our chat centered around food, health, or meal plans."

    except:
        reply = "Oops! Something went wrong while processing. Please try again in a bit."

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)
