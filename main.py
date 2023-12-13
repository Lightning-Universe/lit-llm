from pprint import pprint

import lit


def main():
    # model = lit.LLM("microsoft/phi-1_5")
    model = lit.LLM("mistralai/Mistral-7B-Instruct-v0.1")

    with model.chat(temperature=0.2) as chat:
        chat.generate(prompt="What do you think about pineapple pizza?")
        chat.generate(prompt="Do you think it's better than pepperoni?")

    alpaca = model.prepare_dataset("alpaca")
    # To skip preparation, just create the get dataset directly:
    # alpaca = model.get_dataset("alpaca")

    finetuned = model.finetune(dataset=alpaca, max_iter=512)

    print("Finetuning hyperparameters:")
    pprint(finetuned.hparams)

    with finetuned.chat(temperature=0.2) as chat:
        chat.generate(prompt="What do you think about pineapple pizza?")
        chat.generate(prompt="Do you think it's better than pepperoni?")

    # finetuned.serve(port=8000)

    # a server has just spun up for you. Now run
    # python client.py "What do you think about pineapple pizza?"
    # in a separate terminal, or equivalently make a cURL request
    # curl -H "Content-Type: application/json" -H "X-API-KEY: 1234567890" -X POST -d '{"prompt":"What do you think about pineapple pizza?", "temperature": 0.2}' 127.0.0.1:8000/chat


if __name__ == "__main__":
    main()
