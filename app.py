# app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering , T5ForConditionalGeneration
import torch

# Set your Hugging Face access token
HUGGINGFACE_TOKEN = "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"  # Replace with your actual token

# Load the pre-trained model and tokenizer
model_name = "chitchat00/chitchat-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGINGFACE_TOKEN)
model = T5ForConditionalGeneration.from_pretrained(model_name, use_auth_token=HUGGINGFACE_TOKEN)

# Define a function to generate text
def generate_text(input_text, min_length=10, max_length=100, temperature=0.7, top_p=0.9, repetition_penalty=1.2):  # Set min_length to desired value
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            min_length=min_length,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

min_length = st.selectbox("Minimum Length of Answer:", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150])
max_length = st.selectbox("Maximum Length of Answer:", [50, 100, 150, 200])
temperature = st.selectbox("Temperature:", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
top_p = st.selectbox("Top P:", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
repetition_penalty = st.selectbox("Repetition Penalty:", [1.0, 1.1, 1.2, 1.3, 1.4, 1.5])


# Streamlit UI
st.title("Chitchat Text Generation")
input_text = st.text_area("Enter your text:", "What are the symptoms of diabetes?")
if st.button("Generate"):
    output_text = generate_text(input_text)
    st.write("Output:", output_text)
