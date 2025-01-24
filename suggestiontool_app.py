import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Constants
FINETUNE_JSONL_PATH = "C:\\Users\\spatt\\Desktop\\finetune_streamlit\\data\\finetune.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"

@st.cache_data
def load_finetune_completions(jsonl_path):
    """
    Load unique completions from finetune.jsonl and return as a sorted list.
    """
    completions = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            completions.add(entry["completion"])
    return sorted(completions)

@st.cache_resource
def load_model(model_name):
    """
    Load and cache the SentenceTransformer model.
    """
    return SentenceTransformer(model_name)

def suggest_completions_semantically(profession, completion_list, model, num_suggestions=10):
    """
    Suggest completions for the given profession using semantic similarity.
    """
    # Generate embeddings for the input profession and all completion values
    profession_embedding = model.encode(profession, convert_to_tensor=True)
    completion_embeddings = model.encode(completion_list, convert_to_tensor=True)

    # Compute cosine similarity
    similarities = util.cos_sim(profession_embedding, completion_embeddings).squeeze(0).cpu().numpy()

    # Rank the completions by similarity
    top_indices = np.argsort(similarities)[::-1][:num_suggestions]
    suggestions = [completion_list[i] for i in top_indices]
    return suggestions

def main():
    # Load model and finetune data
    st.title("Profession to Completion Mapper (Semantic Matching)")
    st.markdown(
        """
        **Instructions**:
        - Specify the number of suggestions you want.
        - Enter a profession in the text box.
        - Click "Suggest Completions" to see semantic matches from the classification scheme.
        - Select the best match, and click "Add Mapping."
        - Export your mappings to `prematch.jsonl` when you're done.
        - Use the delete buttons to remove mappings from the list.
        """
    )

    # Safeguard: Initialize session state variables
    if "mappings" not in st.session_state:
        st.session_state.mappings = []

    if "suggestions" not in st.session_state:
        st.session_state.suggestions = []

    if "selected_completion" not in st.session_state:
        st.session_state.selected_completion = None

    # Load model and completions
    with st.spinner("Loading model..."):
        model = load_model(MODEL_NAME)

    with st.spinner("Loading finetune.jsonl..."):
        completion_list = load_finetune_completions(FINETUNE_JSONL_PATH)

    # Input for number of suggestions
    num_suggestions = st.number_input(
        "Number of suggestions:",
        min_value=1,
        max_value=20,
        value=10,
        step=1,
    )

    # Input section for profession
    profession = st.text_input("Enter a profession:", "")

    # Suggest completions
    if st.button("Suggest Completions"):
        if not profession.strip():
            st.error("Please enter a profession to get suggestions.")
        else:
            with st.spinner("Finding suggestions..."):
                suggestions = suggest_completions_semantically(profession, completion_list, model, int(num_suggestions))
            st.session_state.suggestions = suggestions  # Store suggestions in session state
            st.session_state.selected_completion = suggestions[0] if suggestions else None

    # Display suggested completions
    if st.session_state.suggestions:
        st.write("**Suggested Completions:**")
        st.session_state.selected_completion = st.radio(
            "Select the best match:",
            st.session_state.suggestions,
            index=st.session_state.suggestions.index(st.session_state.selected_completion)
            if st.session_state.selected_completion in st.session_state.suggestions
            else 0,
        )

    # Add Mapping
    if st.session_state.selected_completion and profession.strip():
        if st.button("Add Mapping"):
            # Add to the session state mappings
            st.session_state.mappings.append(
                {"prompt_occupation": profession.strip(), "completion": st.session_state.selected_completion}
            )
            st.success(f"Mapping added: {profession.strip()} -> {st.session_state.selected_completion}")
            st.session_state.suggestions = []  # Clear suggestions after adding
            st.session_state.selected_completion = None  # Reset selection

    # Display current mappings with delete buttons
    st.subheader("Current Mappings")
    if st.session_state.mappings:
        for i, mapping in enumerate(st.session_state.mappings):
            col1, col2 = st.columns([6, 1])
            with col1:
                st.write(f"- **{mapping['prompt_occupation']}** → {mapping['completion']}")
            with col2:
                if st.button("Delete", key=f"delete_{i}"):
                    st.session_state.mappings.pop(i)  # Remove the mapping
                    st.experimental_rerun()  # Refresh the app to update the UI
    else:
        st.info("No mappings added yet.")

    # Export mappings
    if st.button("Export to JSONL"):
        if not st.session_state.mappings:
            st.error("No mappings to export.")
        else:
            # Convert mappings to JSONL string
            jsonl_data = "\n".join(json.dumps(mapping) for mapping in st.session_state.mappings)
            st.download_button(
                label="Download prematch.jsonl",
                data=jsonl_data,
                file_name="prematch.jsonl",
                mime="application/jsonl",
            )

if __name__ == "__main__":
    main()
