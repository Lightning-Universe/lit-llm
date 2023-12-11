# LLM Finetune Quest

This studio lets you fine-tune a Phi 1.5 base model using instruction-tuning data.

Phi 1.5 is a 1.3B parameter model created by Microsoft ([technical report](https://www.microsoft.com/en-us/research/publication/textbooks-are-all-you-need-ii-phi-1-5-technical-report/)). Phi 1.5 is competitive with many larger models up to 10B in size.

## Usage

Take a look at `main.py` for an example of finetuning and generation. The steps are as follows.

### Load the base model

Create an instance of the model passing the model name as an argument:

```python
model = llm.LLM("microsoft/phi-1_5")
```

### Chat with the base model

Start a chat and send a prompt to see how the base model behaves:

```python
with model.chat(temperature=0.2) as chat:
    response = chat.generate(prompt="What do you think about pineapple pizza?")
```

### Prepare the dataset

Download and prepare the instruction-tuning dataset. To prepare the [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) dataset call the `prepare_dataset` method of `model`:

```python
alpaca = model.prepare_dataset("alpaca")
```

Once you download and prepare the dataset once, you can create the dataset directly

```python
alpaca = llm.Dataset(name="alpaca", model_name=model)
```

You can also prepare the [Dolly](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) dataset:

```python
alpaca = llm.Dataset(name="dolly", model_name=model)
```

You can also bring your own CSV, in which case you can use (`dataset="csv"`).

```python
mydataset = model.prepare_csv_dataset("mydataset", csv_path="<path_to_csv>")
```

In the latter case, you need to provide a CSV file with the following 3 columns
```
instruction input output
```

and pass it as the `csv_path=<data.csv>` argument to the function.

### Fine-tune the base Phi 1.5 model on the dataset

You can now fine-tune your model on the data. Finetuning will automatically run across all available GPUs.

To finetune, call the `finetune` method on the `model`, and pass the `dataset` that you prepared previously.

```python
finetuned  = model.finetune(dataset=alpaca, max_iter=100)
```

You can pass a number of hyperparameters to `finetune` in order to control the outcome.

### Chat with the model

You can chat with the resulting model just like previously, only creating the chat context using `finetuned`:

```python
with finetuned.chat(temperature=0.2) as chat:
    response = chat.generate(prompt="What do you think about pineapple pizza?")
```

### Start an API inference server

You can serve each model through an API server this way

```python
finetuned.serve(port=8000)
```

You can send a request to the server using

```bash
python client.py "What do you think about pineapple pizza?"
```

in a separate terminal, or equivalently make a cURL request

```bash
curl -H "Content-Type: application/json" -H "X-API-KEY: 1234567890" -X POST -d '{"prompt":"What do you think about pineapple pizza?", "temperature": 0.2}' 127.0.0.1:8000/chat
```