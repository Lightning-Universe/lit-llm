from pathlib import Path
import sys


def prepare_dataset(
    model_name: str = "microsoft/phi-1_5",
    dataset: str = "",
    csv_path: str = "",
    data_dir: str = ""
):
    checkpoint_dir = Path.home() / "checkpoints" / model_name
    csv_path = Path(csv_path)
    data_dir = Path(data_dir)

    if not dataset:
        print("Please provide the name of the dataset (--dataset alpaca | dolly | csv).")

    if dataset == "alpaca":
        from llm.datasets import prepare_alpaca
        prepare_alpaca.prepare(
            checkpoint_dir=checkpoint_dir,
            destination_path=data_dir,
        )

    elif dataset == "dolly":
        from llm.datasets import prepare_dolly
        prepare_dolly.prepare(
            checkpoint_dir=checkpoint_dir,
            destination_path=data_dir,
        )

    elif dataset == "csv":
        from llm.datasets import prepare_csv
        if not csv_path:
            print("Please provide a CSV file with fine-tuning data as the csv_path argument.")
            print("The CSV file must contain three columns: instruction, input and output.")
            return
        prepare_csv.prepare(
            checkpoint_dir=checkpoint_dir,
            csv_path=csv_path,
            destination_path=data_dir,
        )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare_dataset)
