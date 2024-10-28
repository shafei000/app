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
def generate_text(input_text, min_length=50, max_length=100):  # Set min_length to desired value
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(input_ids, min_length=min_length, max_length=max_length)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Streamlit UI
st.title("Chitchat Text Generation")
input_text = st.text_area("Enter your text:", "What are the symptoms of diabetes?")
if st.button("Generate"):
    output_text = generate_text(input_text)
    st.write("Output:", output_text)
