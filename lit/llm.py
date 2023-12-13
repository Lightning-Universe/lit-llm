from contextlib import contextmanager
from pathlib import Path
import shutil
from typing import Optional

from lit import download_model, finetune, prepare_dataset, setup_chat, chat, serve


class Dataset:
    def __init__(self, name, model_name):
        self.name = name
        self.model_name = model_name

    @property
    def data_dir(self):
        return Path("data") / self.name

    def prepare(self):
        if not self.name in ["alpaca", "dolly"]:
            raise ValueError(f"Unsupported dataset {self.name}.")
        # TODO: clean dataset directory
        prepare_dataset(
            model_name=self.model_name,
            dataset=self.name,
            data_dir=self.data_dir
        )
    
    def prepare_csv(self, csv_path: str = ""):
        # TODO: clean dataset directory
        prepare_dataset(
            model_name=self.model_name,
            dataset="csv",
            csv_path=csv_path,
            data_dir=self.data_dir
        )


class Chat:
    def __init__(self, llm, config):
        self.llm = llm
        self.config = config
        self.context = []
    
    def generate(self, prompt="", temperature=None, do_print=True):
        config = self.config.copy()
        if temperature is not None:
            config["temperature"] = temperature
        response = []
        if do_print:
            print(f">>> {prompt}")
        for word in chat(**config, prompt=prompt, context=self.context, log_toks=not do_print):
            if do_print:
                print(word, end="", flush=True)
            response.append(word)
        if do_print:
            print()
        return "".join(response)

    def stream(self, prompt="", temperature=None):
        config = self.config.copy()
        if temperature is not None:
            config["temperature"] = temperature
        return chat(**config, prompt=prompt, context=self.context)

    def generate_with_context(self, context=[], temperature=None):
        config = self.config.copy()
        if temperature is not None:
            config["temperature"] = temperature
        usage = {}
        text = "".join(chat(**config, prompt="", context=context, usage=usage))
        return text, usage


class LLM:
    def __init__(self, model_name):
        self.model_name = model_name
        self.checkpoint_dir = Path("checkpoints") / model_name
        self.checkpoint = "lit_model.pth"
        self.dtype = "bfloat16"
        self.hparams = None

        if not self.checkpoint_dir.exists():
            self.download()

    def download(self):
        download_model(
            model_name=self.model_name,
            checkpoint_dir=self.checkpoint_dir,
            dtype=self.dtype
        )

    def prepare_dataset(self, dataset: str):
        if not dataset in ["alpaca", "dolly"]:
            raise ValueError(f"Unsupported dataset {dataset}. Use `prepare_csv_dataset` to add custom datasets.")
        dataset = Dataset(name=dataset, model_name=self.model_name)
        dataset.prepare()
        return dataset
    
    def prepare_csv_dataset(self, dataset: str, csv_path: str):
        dataset = Dataset(name=dataset, model_name=self.model_name)
        dataset.prepare_csv(csv_path=csv_path)
        return dataset
    
    def get_dataset(self, dataset: str):
        dataset = Dataset(name=dataset, model_name=self.model_name)
        return dataset

    def finetune(self,
        dataset,
        max_iter=1000,
        batch_size: int = 128,
        micro_batch_size: int = 4,
        n_epochs: int = 3,
        learning_rate: float = 3e-4,
        max_seq_length: Optional[int] = 1024,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        eval_interval: int = 100,
        save_interval: int = 100,
        eval_iters: int = 100,
        log_interval: int = 1,
    ):
        # TODO: add "mode" argument to provide sets of defaults

        prefix = f"{self.model_name}-{dataset.name}-"
        existing_dirs = self.checkpoint_dir.parent.parent.glob(f"{prefix}*")
        existing_dirs = [el for el in existing_dirs if el.name[-4:].isnumeric()]
        if existing_dirs:
            existing_dirs.sort()
            last_dir = existing_dirs[-1]
            last_suffix = int(last_dir.name[-4:])
            suffix = f"{last_suffix + 1:04}"
        else:
            suffix = f"{0:04}"
        finetuned_model_name = f"{prefix}{suffix}"

        out_dir = Path("out") / self.model_name
        out_checkpoint_dir = self.checkpoint_dir.parent.parent / finetuned_model_name
        out_checkpoint_dir.mkdir(exist_ok=True, parents=True)

        files_to_copy = [el for el in self.checkpoint_dir.iterdir() if el != "lit_model.pth"]
        for el in files_to_copy:
            shutil.copy(el, out_checkpoint_dir)

        out_checkpoint = out_checkpoint_dir / self.checkpoint

        hparams = finetune(
            model_name=self.model_name,
            data_dir=dataset.data_dir,
            out_dir=out_dir,
            out_checkpoint=out_checkpoint,
            max_iter=max_iter,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            max_seq_length=max_seq_length,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            eval_interval=eval_interval,
            save_interval=save_interval,
            eval_iters=eval_iters,
            log_interval=log_interval,
        )

        hparams.save(out_checkpoint_dir)

        finetuned = LLM(model_name=finetuned_model_name)
        finetuned.checkpoint_dir = out_checkpoint_dir
        finetuned.dtype = self.dtype
        finetuned.hparams = hparams
        return finetuned
    
    @contextmanager
    def chat(self, temperature=0.2):
        # if model is not set up for chat, set it up
        chat_config = setup_chat(
            model_name=self.model_name,
            checkpoint=self.checkpoint_dir / self.checkpoint,
            temperature=temperature,
            quantize="bnb.nf4-dq",
            precision="bf16-true",
        )

        chat = Chat(self, chat_config)
        try:
            yield chat
        finally:
            chat = None
            # gc

    def serve(self, temperature=0.2, device_ids=[0], port=8000, timeout_keep_alive=30, blocking=True):
        serve(
            llm=self,
            temperature=temperature,
            device_ids=device_ids,
            port=port,
            timeout_keep_alive=timeout_keep_alive,
            blocking=blocking
        )
