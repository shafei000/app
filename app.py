import streamlit as st
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

# Set your Hugging Face access token
HUGGINGFACE_TOKEN = "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"  # Replace with your actual token

# Load the pre-trained model and tokenizer
model_name = "chitchat00/chitchat-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGINGFACE_TOKEN)
model = T5ForConditionalGeneration.from_pretrained(model_name, use_auth_token=HUGGINGFACE_TOKEN)

# Define a function to generate text
def generate_text(input_text, min_length=10, max_length=100, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
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

# Streamlit UI
st.title("Chitchat Text Generation")

# Text area for input
input_text = st.text_area("Enter your text:", "What are the symptoms of diabetes?")

# Button to toggle slider visibility
if st.button("Toggle Settings"):
    st.session_state.show_sliders = not st.session_state.get("show_sliders", False)

# Show sliders if the toggle is active
if st.session_state.get("show_sliders", False):
    min_length = st.slider("Minimum Length of Answer:", 10, 150, 50)  # Slider for minimum length
    max_length = st.slider("Maximum Length of Answer:", 50, 200, 100)  # Slider for maximum length
    temperature = st.slider("Temperature:", 0.0, 1.0, 0.7)  # Slider for temperature
    top_p = st.slider("Top P:", 0.0, 1.0, 0.9)  # Slider for top_p
    repetition_penalty = st.slider("Repetition Penalty:", 1.0, 2.0, 1.2)  # Slider for repetition penalty
else:
    # Set default values for hidden sliders
    min_length = 10
    max_length = 100
    temperature = 0.7
    top_p = 0.9
    repetition_penalty = 1.2

if st.button("Generate"):
    output_text = generate_text(input_text, min_length=min_length, max_length=max_length,
                                 temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)
    st.write("Output:", output_text)
