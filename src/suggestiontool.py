import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util  # For semantic similarity

# Load the unique completion values from finetune.jsonl
def load_finetune_completions(jsonl_path):
    completions = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            completions.add(entry["completion"])
    return sorted(completions)  # Return a sorted list of unique completions


# Semantic similarity-based suggestion
def suggest_completions_semantically(profession, completion_list, model, num_suggestions=5):
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


# Interface for mapping professions
def map_professions_semantically(completion_list, model):
    """
    A GUI for mapping professions to completions using semantic similarity. Outputs results to JSONL format.
    """
    # List to store the results
    results = []

    # Function to handle the "Submit" button
    def submit_mapping():
        nonlocal results
        # Get user inputs
        profession = profession_input.get().strip()
        selected_completion = completion_var.get().strip()

        if not profession:
            messagebox.showerror("Error", "Please enter a profession.")
            return

        if not selected_completion or selected_completion == "Select a value":
            messagebox.showerror("Error", "Please select a completion value.")
            return

        # Append the mapping to results
        results.append({"prompt_occupation": profession, "completion": selected_completion})
        messagebox.showinfo("Success", f"Mapping saved: {profession} -> {selected_completion}")
        # Clear inputs for the next entry
        profession_input.delete(0, tk.END)
        completion_var.set("Select a value")

    # Function to handle the "Save to File" button
    def save_to_file():
        if not results:
            messagebox.showerror("Error", "No mappings to save.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Mappings as JSONL",
            defaultextension=".jsonl",
            filetypes=(("JSONL Files", "*.jsonl"), ("All Files", "*.*")),
        )
        if not save_path:
            return

        # Write results to the selected file
        with open(save_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        messagebox.showinfo("Success", f"Mappings saved to {save_path}")

    # Function to handle completion suggestions
    def suggest_values():
        profession = profession_input.get().strip()
        if not profession:
            messagebox.showerror("Error", "Please enter a profession to get suggestions.")
            return

        suggestions = suggest_completions_semantically(profession, completion_list, model)
        if suggestions:
            completion_var.set(suggestions[0])  # Set the first suggestion as default
            suggestion_menu["menu"].delete(0, "end")  # Clear the dropdown menu
            for suggestion in suggestions:
                suggestion_menu["menu"].add_command(
                    label=suggestion,
                    command=lambda value=suggestion: completion_var.set(value),
                )
        else:
            messagebox.showinfo("No Suggestions", "No close matches found for the entered profession.")

    # Create the main GUI window
    root = tk.Tk()
    root.title("Profession to Completion Mapper (Semantic)")

    # Labels and inputs
    tk.Label(root, text="Enter Profession:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    profession_input = tk.Entry(root, width=30)
    profession_input.grid(row=0, column=1, padx=10, pady=5)

    tk.Label(root, text="Select Completion:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    completion_var = tk.StringVar(value="Select a value")
    suggestion_menu = tk.OptionMenu(root, completion_var, "Select a value", *completion_list)
    suggestion_menu.grid(row=1, column=1, padx=10, pady=5)

    # Buttons
    suggest_button = tk.Button(root, text="Suggest Completions", command=suggest_values)
    suggest_button.grid(row=2, column=0, padx=10, pady=10)

    submit_button = tk.Button(root, text="Submit Mapping", command=submit_mapping)
    submit_button.grid(row=2, column=1, padx=10, pady=10)

    save_button = tk.Button(root, text="Save to File", command=save_to_file)
    save_button.grid(row=3, column=0, columnspan=2, pady=10)

    root.mainloop()


if __name__ == "__main__":
    # Ask the user to locate finetune.jsonl
    finetune_path = filedialog.askopenfilename(
        title="Select finetune.jsonl",
        filetypes=(("JSONL Files", "*.jsonl"), ("All Files", "*.*")),
    )
    if not finetune_path:
        print("No finetune.jsonl file selected. Exiting.")
        exit()

    # Load unique completions
    completions = load_finetune_completions(finetune_path)
    print(f"Loaded {len(completions)} unique completions from {finetune_path}.")

    # Load a pre-trained sentence transformer model
    print("Loading sentence transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight and fast for sentence embeddings
    print("Model loaded successfully.")

    # Launch the profession-to-completion mapper with semantic matching
    map_professions_semantically(completions, model)
