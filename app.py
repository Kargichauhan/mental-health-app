import streamlit as st
import openai

# Check if API key exists in Streamlit secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OpenAI API Key is missing. Please add it to Streamlit Secrets.")
    st.stop()

# Retrieve API Key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Check if the API key is empty
if not OPENAI_API_KEY:
    st.error("OpenAI API Key is empty. Ensure it's set correctly in Streamlit Secrets.")
    st.stop()

# Initialize OpenAI Client with error handling
try:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {e}")
    st.stop()

# Streamlit UI
st.title("AI Mental Health Counselor")
st.write("Enter your issue, and I will generate guidance.")

user_input = st.text_area("Describe your issue:", "")

if st.button("Get Advice"):
    if user_input.strip():  # Ensures input is not empty
        with st.spinner("Loading response..till then give a smile :)"):
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

                st.subheader("Advice:")
                st.write(response.choices[0].message.content)

            except openai.OpenAIError as e:
                st.error(f"OpenAI API Error: {e}")
            except Exception as e:
                st.error(f"Unexpected Error: {e}")
    else:
        st.warning("Please enter a description before submitting.")
