import os
from pathlib import Path
import shutil
from typing import Optional

import torch
from lightning_utilities.core.imports import RequirementCache

_SAFETENSORS_AVAILABLE = RequirementCache("safetensors")


def collect(
    repo_id: Optional[str] = None, checkpoint_path: Optional[str] = None
) -> None:
    if repo_id is None:
        from lit_gpt.config import configs

        options = [f"{config['org']}/{config['name']}" for config in configs]
        print("Please specify --repo_id <repo_id>. Available values:")
        print("\n".join(options))
        return

    checkpoint_path = Path(checkpoint_path)
    from_safetensors = True if list(checkpoint_path.glob("*.safetensors")) else False

    directory = Path("checkpoints") / repo_id
    directory.mkdir(parents=True, exist_ok=True)

    shutil.copytree(str(checkpoint_path), str(directory), dirs_exist_ok=True)

    # convert safetensors to PyTorch binaries
    if from_safetensors:
        from safetensors import SafetensorError
        from safetensors.torch import load_file as safetensors_load

        print("Converting .safetensor files to PyTorch binaries (.bin)")
        for safetensor_path in directory.glob("*.safetensors"):
            bin_path = safetensor_path.with_suffix(".bin")
            try:
                result = safetensors_load(safetensor_path)
            except SafetensorError as e:
                raise RuntimeError(f"{safetensor_path} is likely corrupted.") from e
            print(f"{safetensor_path} --> {bin_path}")
            torch.save(result, bin_path)
            os.remove(safetensor_path)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(collect)