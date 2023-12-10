from pprint import pprint

import llm


def main():
    model = llm.LLM("microsoft/phi-1_5")

    with model.chat(temperature=0.2) as chat:
        response = chat.generate(prompt="What do you think about pineapple pizza?")

    alpaca = model.prepare_dataset("alpaca")
    # To skip preparation, just create the get dataset directly:
    # alpaca = model.get_dataset("alpaca")

    finetuned  = model.finetune(dataset=alpaca, max_iter=100)

    pprint(finetuned.hparams)

    with finetuned.chat(temperature=0.2) as chat:
        response = chat.generate(prompt="What do you think about pineapple pizza?")


if __name__ == "__main__":
    main()
