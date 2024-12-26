import streamlit as st
import chatbot

#

def main():
   chatbot.chat()


   
# for user (You)
st.markdown("""
    <style>
    .user-label {
        color: #0066ff;
        font-weight: bold;
        font-size: 19px;
    }
    </style>
""", unsafe_allow_html=True)

# for Assistant
st.markdown("""
    <style>
    .assistant-label {
        color: #FF4B4B;
        font-weight: bold;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()