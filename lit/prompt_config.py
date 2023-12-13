from functools import partial
from pathlib import Path
import re


def create_system_prompt(
        system_b, system_e,
        user_b, user_e,
        assistant_b, assistant_e,
        sys_in_user=False,
        default_system_message=""):
    
    def system_prompt(prompt, context=[]):
        out = []
        if default_system_message:
            if sys_in_user:
                out.append(user_b)
            out.append(system_b)
            out.append(default_system_message)
            out.append(system_e)

        for message in context:
            if message["role"] == "system":
                if default_system_message:
                    out[1] = message["content"]
                else:
                    out.append(system_b)
                    out.append(message["content"])
                    out.append(system_e)

            if message["role"] == "user":
                if out and out[-1] != system_e:
                    out.append(user_b)
                out.append(message["content"])
                out.append(user_e)

            if message["role"] == "assistant":
                out.append(assistant_b)
                out.append(message["content"])
                out.append(assistant_e)

        if prompt:
            out.append(user_b)
            out.append(prompt)
            out.append(user_e)

        out.append(assistant_b)

        return "".join(out)

    return system_prompt


def prompt_config(checkpoint_dir, tokenizer):
    checkpoint_name = str(checkpoint_dir)
    if re.search(r"stabilityai.*tuned-alpha", checkpoint_name):
        # system_prompt = (
        #     "<|SYSTEM|># StableLM Tuned (Alpha version)\n- StableLM is a helpful and harmless open-source AI language"
        #     " model developed by StabilityAI.\n- StableLM is excited to be able to help the user, but will refuse to do"
        #     " anything that could be considered harmful to the user.\n- StableLM is more than just an information"
        #     " source, StableLM is also able to write poetry, short stories, and make jokes.\n- StableLM will refuse to"
        #     " participate in anything that could harm a human.<|USER|>{prompt}<|ASSISTANT|>"
        # )
        system_prompt = create_system_prompt_("<|SYSTEM|>", "", "<|USER|>", "", "<|ASSISTANT|>", "",
            default_system_message="># StableLM Tuned (Alpha version)\n- StableLM is a helpful and harmless open-source AI language"
                " model developed by StabilityAI.\n- StableLM is excited to be able to help the user, but will refuse to do"
                " anything that could be considered harmful to the user.\n- StableLM is more than just an information"
                " source, StableLM is also able to write poetry, short stories, and make jokes.\n- StableLM will refuse to"
                " participate in anything that could harm a human.")
        stop_tokens = (
            [tokenizer.eos_id],
            [tokenizer.token_to_id("<|SYSTEM|>")],
            [tokenizer.token_to_id("<|ASSISTANT|>")],
            [tokenizer.token_to_id("<|USER|>")],
        )
        return system_prompt, stop_tokens
    if re.search(r"togethercomputer.*Chat", checkpoint_name):
        # system_prompt = "<human>: {prompt}\n<bot>:"
        system_prompt = create_system_prompt("", "", "<human>: ", "\n", "<bot>", "\n")
        lt, gt = tokenizer.token_to_id("<"), tokenizer.token_to_id(">:")
        stop_tokens = (
            [tokenizer.eos_id],
            # annoyingly, there's no single stop token for these
            [lt, tokenizer.token_to_id("human"), gt],
            [lt, tokenizer.token_to_id("bot"), gt],
        )
        return system_prompt, stop_tokens
    if re.search(r"togethercomputer.*Instruct", checkpoint_name):
        # system_prompt = "Q: {prompt}\nA:"
        system_prompt = create_system_prompt("", "", "Q: ", "\n", "A: ", "\n")
        colon = tokenizer.token_to_id(":")
        stop_tokens = (
            [tokenizer.eos_id],
            # annoyingly, there's no single stop token for these
            [tokenizer.token_to_id("Q"), colon],
            [tokenizer.token_to_id("Question")],
            [tokenizer.token_to_id("A"), colon],
            [tokenizer.token_to_id("Label"), colon],
            [187, 187],  # '\n', '\n'
            [535],  # '\n\n'
            [2756],  # '\n\n\n'
        )
        return system_prompt, stop_tokens
    if re.search(r"falcon.*-instruct", checkpoint_name):
        # First line could be modified. AFAIK Falcon doesn't impose a specific system prompt
        # The instruction to not prefix its replies doesn't work always, but better than nothing
        # system_prompt = "Do not prefix your replies with 'Bot: '\nUser: {prompt}\n"
        system_prompt = create_system_prompt("", "", "User: ", "\n", "", "\n",
            default_system_message="Do not prefix your replies with 'Bot: '\n")

        # I've also tried just "{prompt}\n" but the model seems to ramble more often
        stop_tokens = (
            [tokenizer.eos_id],
            # the model rarely emits the eos token and instead outputs newlines, but we cannot use them
            # to stop or else things like code generation wouldn't work
            [tokenizer.token_to_id("User"), tokenizer.token_to_id(":")],
            [193, tokenizer.token_to_id("User")],  # 193: '\n'
        )
        return system_prompt, stop_tokens
    if re.search(r"vicuna|longchat", checkpoint_name):
        # https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md#prompt-template
        # system_prompt = (
        #     "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, "
        #     "detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:"
        # )
        system_prompt = create_system_prompt("", "", "USER: ", " ", "ASSISTANT: ", " ",
            default_system_message="A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions. ")
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens
    if re.search("Llama-2.*-chat", checkpoint_name):
        # b_inst, e_inst = "[INST]", "[/INST]"
        # b_sys, e_sys = "<<SYS>>\n", "\n<</SYS>>\n\n"
        # system_prompt = (
        #     f"{b_inst} {b_sys}You are a helpful, respectful and honest assistant. Always answer as helpfully as"
        #     " possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist,"
        #     " toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and"
        #     " positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why"
        #     " instead of answering something not correct. If you don't know the answer to a question, please don't"
        #     f" share false information.{e_sys} {{prompt}} {e_inst} "
        # )
        system_prompt = create_system_prompt(" <<SYS>>", "<</SYS>> ", "[INST]", "[/INST]", "", "", sys_in_user=True,
            default_system_message="You are a helpful, respectful and honest assistant. Always answer as helpfully as"
            " possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist,"
            " toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and"
            " positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why"
            " instead of answering something not correct. If you don't know the answer to a question, please don't"
            " share false information.")
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    if re.search("FreeWilly2", checkpoint_name):
        # system_prompt = (
        #     "### System:\nThis is a system prompt, please behave and help the user.\n\n"
        #     "### User:\n"
        #     "{prompt}\n\n"
        #     "### Assistant:\n"
        # )
        system_prompt = create_system_prompt("### System:\n", "\n\n", "### User:\n", "\n\n", "### Assistant:\n", "\n\n",
            default_system_message="This is a system prompt, please behave and help the user.")
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    if re.search("Platypus", checkpoint_name):
        # system_prompt = "### Instruction:\n\n{prompt}\n\n### Response:\n"
        system_prompt = create_system_prompt("", "", "### Instruction:\n\n", "\n\n", "### Response:\n", "\n\n")
        # this checkpoint doesn't emit the eos token very consistently
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    if re.search("NousResearch", checkpoint_name):
        # system_prompt = "### Instruction:\n{prompt}\n\n### Response:\n"
        system_prompt = create_system_prompt("", "", "### Instruction:\n", "\n\n", "### Response:\n", "\n\n")
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    if re.search("stablecode-instruct", checkpoint_name):
        # system_prompt = "###Instruction\n{prompt}###Response\n"
        system_prompt = create_system_prompt("", "", "###Instruction:\n", "", "###Response:\n", "")
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    if re.search("CodeLlama|Mistral.*Instruct", checkpoint_name):
        # for CodeLLama, we don't set a default system prompt, but it is supported:
        # https://huggingface.co/blog/codellama#conversational-instructions
        # Mistral does not: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1#instruction-format
        # b_inst, e_inst = "<s>[INST]", "[/INST]"
        # system_prompt = f"{b_inst} {{prompt}} {e_inst}"
        system_prompt = create_system_prompt("", "", "<s>[INST]", "[/INST]", "", "")
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    if re.search("phi", checkpoint_name):
        # system_prompt = "{prompt}\n\nAnswer:"
        system_prompt = create_system_prompt("", "", "", "\n", "Answer:", "")

        stop_tokens = (
            [tokenizer.eos_id],
            # [tokenizer.token_to_id("Answer"), tokenizer.token_to_id(":")],
            # [198, tokenizer.token_to_id("Answer"), tokenizer.token_to_id(":")],
            # the model rarely emits the eos token and instead outputs newlines, but we cannot use them
            # to stop when we use the model for code generation
            [198, 198],  # '\n', '\n'
        )
        return system_prompt, stop_tokens

    if re.search(r"TinyLlama.*Chat", checkpoint_name):
        # system_prompt = (
        #     "<|system|>\n"
        #     "You are a friendly chatbot who always gives helpful, detailed, and polite answers.</s>\n"
        #     "<|user|>\n"
        #     "{prompt}</s>\n"
        #     "<|assistant|>\n"
        # )
        system_prompt = create_system_prompt("<|system|>\n", "</s>\n", "<|user|>\n", "</s>\n", "<|assistant|>\n", "</s>\n")
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    # default format
    return create_system_prompt("", "", "", "", "", ""), ([tokenizer.eos_id],)
