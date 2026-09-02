from huggingface_hub import snapshot_download
import os

model_id = "EleutherAI/pythia-1.4b"
revisions = ["step1000", "step8000", "step32000", "step72000"]
output_dir = "./pythia_revisions"
os.makedirs(output_dir, exist_ok=True)

for revision in revisions:
    print(f"Downloading {model_id} at revision {revision}...")
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        cache_dir=output_dir,
        local_dir=f"{output_dir}/pythia-1.4b-{revision}"
    )
    print(f"✓ Downloaded {revision}")

print("\n✓ All revisions downloaded successfully!")
