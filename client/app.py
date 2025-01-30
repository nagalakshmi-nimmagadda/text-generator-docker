import streamlit as st
import requests

st.title("SmolLM2-135M Shakespeare Text Generator")

st.write("""
This is a 135M parameter language model trained on Shakespeare's text.
Enter a prompt and adjust the generation parameters to see the model's output.
""")

# Input form
with st.form("text_generation_form"):
    prompt = st.text_area(
        "Enter your prompt",
        value="To be or not to be",
        height=100
    )
    
    col1, col2 = st.columns(2)
    with col1:
        max_tokens = st.slider(
            "Max Tokens",
            min_value=10,
            max_value=200,
            value=50,
            step=10
        )
    with col2:
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.1
        )
    
    submit = st.form_submit_button("Generate Text")

if submit:
    with st.spinner("Generating text..."):
        try:
            response = requests.post(
                'http://model-server:8000/generate',
                json={
                    'prompt': prompt,
                    'max_tokens': max_tokens,
                    'temperature': temperature
                }
            )
            if response.status_code == 200:
                st.success("Text generated! View results at: http://localhost:8000")
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Error connecting to model server: {str(e)}")

# Example prompts
st.write("### Example Prompts")
example_prompts = [
    "My bounty is as boundless as the sea,",
    "All the world's a stage,",
    "The course of true love never did",
    "We are such stuff as dreams"
]

for prompt in example_prompts:
    if st.button(prompt):
        with st.spinner("Generating text..."):
            try:
                response = requests.post(
                    'http://model-server:8000/generate',
                    json={
                        'prompt': prompt,
                        'max_tokens': 50,
                        'temperature': 0.7
                    }
                )
                if response.status_code == 200:
                    st.success("Text generated! View results at: http://localhost:8000")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Error connecting to model server: {str(e)}") 