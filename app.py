import streamlit as st
import openai

# 🔹 Set your OpenAI API Key
OPENAI_API_KEY = "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"  # Replace with your actual API key

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)  # 🔹 Create OpenAI client

st.title(" AI Mental Health Counselor")
st.write("Enter issue, and I will generate guidance :) ")

user_input = st.text_area("Describe your issue:", "")

if st.button("Legacy Advice: "):
    if user_input:
        with st.spinner("Loading response..till then give a smile."):
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": """
                        You are an experienced mental health counselor providing compassionate, evidence-based support. 
                        Your goal is to offer guidance that is empathetic, non-judgmental, and solution-oriented.
                        """},
                        {"role": "user", "content": user_input}
                    ]
                )

                st.subheader("Legacy Advices:")
                st.write(response.choices[0].message.content)

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("⚠️ Please enter a description.")
