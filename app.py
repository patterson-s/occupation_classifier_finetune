import streamlit as st
from pathlib import Path
import json
import openai
from typing import Optional, Dict, Tuple

# Constants
MAX_REQUESTS_PER_SESSION = 5
BASE_DIR = Path(__file__).parent
DECODER_PATH = BASE_DIR / "data/finetune.jsonl"
MODEL_ID = "ft:gpt-3.5-turbo-0613:personal::7qnGb8rm"
CONTACT_INFO = "scott.patterson[at]mail[dot]mcgill[dot]ca"

def load_decoder() -> Dict[str, str]:
    with open(DECODER_PATH, "r") as f:
        return {json.loads(line)["transformed_completion"].strip(): 
                json.loads(line)["completion"] for line in f}

def get_classification(occup_title: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "classify this entry:"},
                {"role": "user", "content": occup_title}
            ],
            max_tokens=50,
            temperature=0.1
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.OpenAIError as e:
        return f"Error: {str(e)}"

def decode_classification(raw_output: str, decoder_map: Dict[str, str]) -> Optional[str]:
    return decoder_map.get(raw_output)

def initialize_session_state():
    if 'request_count' not in st.session_state:
        st.session_state.request_count = 0

def check_rate_limit() -> Tuple[bool, str]:
    if st.session_state.request_count >= MAX_REQUESTS_PER_SESSION:
        message = (f"You've reached the maximum number of requests ({MAX_REQUESTS_PER_SESSION}). "
                  f"To use this model in your work, please contact {CONTACT_INFO}")
        return False, message
    st.session_state.request_count += 1  # Move the increment here
    return True, ""

def main():
    # Setup
    initialize_session_state()
    
    # Configure OpenAI
    openai.api_key = st.secrets["openai"]["api_key"]
    
    # Load decoder
    decoder_map = load_decoder()

    st.title("Occupation Classifier")
    st.markdown("""
    **Fine-Tuned GPT-3.5 Model**  
    Enter an occupation title, and the model will classify it according to Bureau of Labor Statistics codes.
    """)

    # Display remaining requests
    st.info(f"Remaining requests: {MAX_REQUESTS_PER_SESSION - st.session_state.request_count}")

    user_input = st.text_input("Enter an occupation title:", "")
    
    if st.button("Classify") and user_input.strip():
        can_proceed, message = check_rate_limit()
        
        if not can_proceed:
            st.warning(message)
            return

        with st.spinner("Classifying..."):
            raw_classification = get_classification(user_input)
            human_readable = decode_classification(raw_classification, decoder_map)

        st.subheader("Results")
        st.write(f"**Raw Classification**: {raw_classification}")
        
        if human_readable:
            st.write(f"**Human-Readable Classification**: {human_readable}")
        else:
            st.warning("No match found for the raw output. Likely hallucination.")

if __name__ == "__main__":
    main()