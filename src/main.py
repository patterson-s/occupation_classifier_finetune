import sys
import pandas as pd

from pre_matching import load_pre_match_dict, pre_match_occupation

# We will use Tkinter to prompt the user for file paths.
try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    print("Tkinter is not available. Please install or use an environment that supports Tkinter.")
    sys.exit(1)

def select_file_dialog(title="Select File", filetypes=(("All Files", "*.*"),)):
    """
    Opens a file dialog for the user to select a file.
    Returns the path to the selected file or None if no file is selected.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes
    )
    root.destroy()
    return file_path if file_path else None


def select_save_dialog(title="Save File As...", filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))):
    """
    Opens a file dialog for the user to select a save location.
    Returns the path to the selected file or None if no file is selected.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.asksaveasfilename(
        title=title,
        filetypes=filetypes,
        defaultextension=".csv"
    )
    root.destroy()
    return file_path if file_path else None


def run_pre_match_pipeline():
    """
    Guides the user:
     1) to select a CSV file,
     2) automatically loads the JSONL from a known path,
     3) asks which column to use for occupation data,
     4) performs pre_match_occupation,
     5) and allows them to save the output CSV.
    """

    # ---- 1) Prompt for CSV input ----
    print("A file dialog will open. Please select the CSV file containing occupation data.")
    csv_path = select_file_dialog(
        title="Select CSV file with occupation data",
        filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
    )
    if not csv_path:
        print("No CSV file was selected. Exiting.")
        return

    print(f"Selected CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    print("CSV loaded successfully.\n")

    # ---- 2) Load the JSONL from a known path ----
    # Replace with the actual location of your finetune.jsonl
    jsonl_path = "C:/Users/spatt/Desktop/finetune_streamlit/data/finetune.jsonl"

    print(f"Loading finetune.jsonl from known location: {jsonl_path}")
    pre_match_dict = load_pre_match_dict(jsonl_path)
    print("Dictionary (finetune.jsonl) loaded successfully.\n")

    # ---- 3) Ask which column contains occupation data ----
    columns = list(df.columns)
    print("Columns in the CSV:")
    for i, col in enumerate(columns):
        print(f"{i}. {col}")

    column_index = input("\nType the number of the column containing the occupation data: ")
    try:
        column_index = int(column_index)
        if column_index < 0 or column_index >= len(columns):
            raise ValueError
    except ValueError:
        print("Invalid column index. Exiting.")
        return

    occupation_column = columns[column_index]
    print(f"You selected column: '{occupation_column}'\n")

    # ---- 4) Perform the pre-match step ----
    print("Performing dictionary pre-match...")
    df_updated = pre_match_occupation(df, occupation_column, pre_match_dict)
    print("Pre-match completed. Two new columns ('final_output', 'method') have been added.\n")

    # ---- 5) Prompt user to save the updated CSV ----
    print("A file dialog will open for you to choose where to save the updated CSV.")
    save_path = select_save_dialog(
        title="Save Updated CSV",
        filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
    )
    if not save_path:
        print("No output file selected. Exiting without saving.")
        return

    df_updated.to_csv(save_path, index=False)
    print(f"Updated CSV saved to: {save_path}")


if __name__ == "__main__":
    run_pre_match_pipeline()
