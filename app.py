# app.py
import streamlit as st
from transformers import AutoTokenizer, T5ForConditionalGeneration ,AutoModelForQuestionAnswering
import torch

# Load the pre-trained T5 model
model_name = "chitchat00/chitchat0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Define a function to generate text
def generate_text(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(input_ids)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Streamlit UI
st.title("CHitchat Text Generation")
input_text = st.text_area("Enter your text:", "translate English to French: The house is wonderful.")
if st.button("Generate"):
    output_text = generate_text(input_text)
    st.write("Output:", output_text)
