from pathlib import Path

from lit.models import convert_hf_checkpoint, download

def download_model(
    model_name="microsoft/phi-1_5",
    checkpoint_dir=None,
    dtype="bfloat16"
):
    download.download_from_hub(repo_id=model_name)

    if checkpoint_dir is None:
        checkpoint_dir = Path.home() / "checkpoints" / model_name

    convert_hf_checkpoint.convert_hf_checkpoint(checkpoint_dir=checkpoint_dir, dtype=dtype)
