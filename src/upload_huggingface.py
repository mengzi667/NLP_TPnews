from huggingface_hub import HfApi
import os

# --- Configuration ---
# Your local folder path to the best model checkpoint
LOCAL_MODEL_PATH = "./src/models/mac_hyper_search_results/run-11/checkpoint-716"
# The desired name for your repository on the Hugging Face Hub
HUB_MODEL_ID = "Mengzi667/macbert-traffic-intent-classifier"

# --- Main Logic ---
if __name__ == '__main__':
    api = HfApi()

    print(f"--- Preparing to upload model to Hugging Face Hub ---")

    # 1. (NEW STEP) Create the repository on the Hub.
    #    This command ensures the repository exists before you try to upload.
    #    `repo_exist_ok=True` means if the repo already exists, it won't cause an error.
    print(f"Creating repository '{HUB_MODEL_ID}' on the Hub...")
    api.create_repo(
        repo_id=HUB_MODEL_ID,
        repo_type="model",
        exist_ok=True
    )
    print("Repository created or already exists.")

    # 2. Upload the entire folder content.
    print(f"Uploading model from '{LOCAL_MODEL_PATH}'...")
    api.upload_folder(
        folder_path=LOCAL_MODEL_PATH,
        repo_id=HUB_MODEL_ID,
        repo_type="model",
        commit_message="Upload fine-tuned MacBERT model for traffic intent classification"
    )

    print("\n--- Model upload successful! ---")
    print(f"You can now find your model at: https://huggingface.co/{HUB_MODEL_ID}")