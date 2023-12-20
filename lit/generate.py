import sys
import time
from pathlib import Path
from typing import Iterator, List, Literal, Optional, Tuple, Any

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision

from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    load_checkpoint,
)

from lit.prompt_config import prompt_config


def multinomial_num_samples_1(probs: torch.Tensor) -> torch.Tensor:
    if torch._dynamo.is_compiling():
        # Faster alternative to `torch.multinomial(probs, num_samples=1)` that is also CUDAGraph friendly
        distribution = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / distribution, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
    logits = logits[0, -1]
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, i = torch.topk(logits, min(top_k, logits.size(-1)))
        # do not use `torch.where` as in nanogpt because it will repeat top-k collisions
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
    # optionally scale the logits and sample from a probability distribution
    if temperature > 0.0:
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
        return multinomial_num_samples_1(probs)
    return torch.argmax(logits, dim=-1, keepdim=True)


def next_token(model: GPT, input_pos: torch.Tensor, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    logits = model(x, input_pos)
    next = sample(logits, **kwargs)
    return next.type_as(x)


@torch.inference_mode()
def generate(
    model: GPT,
    prompt: torch.Tensor,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    stop_tokens: Tuple[List[int], ...] = (),
) -> Iterator[torch.Tensor]:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as possible.

    Args:
        model: The model to use.
        prompt: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
        stop_tokens: If specified, stop generating any more token once one of this list is generated.
    """
    T = prompt.size(0)
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

    device = prompt.device
    buffer_length = max((len(tokens) for tokens in stop_tokens), default=1)
    yield_i = 0
    input_pos = torch.arange(0, T, device=device)
    tokens = []
    token = prompt
    for t in range(1, max_returned_tokens - T + 1):
        token = next_token(model, input_pos, token.view(1, -1), temperature=temperature, top_k=top_k)
        tokens.append(token)
        # check the stop condition
        if any((l := len(st)) <= len(tokens) and all(a == b for a, b in zip(tokens[-l:], st)) for st in stop_tokens):
            return
        # if the buffer is full
        if t - yield_i >= buffer_length:
            # we know this idx is not part of stop tokens, safe to yield
            yield from tokens[yield_i:t]
            yield_i = t
        input_pos = input_pos[-1:].add_(1)


def decode(fabric: L.Fabric, tokenizer: Tokenizer, token_stream: Iterator[torch.Tensor]) -> int:
    tokens_generated = 0
    if tokenizer.backend == "huggingface":
        try:
            for token in token_stream:
                yield tokenizer.decode(token)
                tokens_generated += 1
        except KeyboardInterrupt:
            # support stopping generation
            return
    elif tokenizer.backend == "sentencepiece":
        # sentencepiece does not support decoding token-by-token because it adds spaces based on the surrounding tokens
        # meaning that we need to decode everything each time
        so_far = torch.tensor([], dtype=torch.long, device=fabric.device)
        decoded_so_far = ""
        try:
            for token in token_stream:
                so_far = torch.cat((so_far, token.view(-1)))
                decoded_new = tokenizer.decode(so_far)
                yield decoded_new[len(decoded_so_far) :]
                decoded_so_far = decoded_new
                tokens_generated += 1
        except KeyboardInterrupt:
            # support stopping generation
            return
    else:
        raise NotImplementedError(tokenizer.backend)
    return


def setup_chat(
    model_name = "microsoft/phi-1_5",
    checkpoint: Optional[str] = None,
    top_k: Optional[int] = 200,
    temperature: float = 0.2,
    quantize: Optional[Literal["nf4"]] = None,
    precision: Optional[str] = None,
    compile: bool = False,
):
    """Starts a conversation with a tuned GPT model.

    Args:
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_dir: The checkpoint directory to load.
        quantize: Whether to quantize the model and using which method:
            - nf4: 4-bit float quantization from bitsandbytes
            - nf4-dq: 4-bit float double quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        precision: Indicates the Fabric precision setting to use.
        compile: Whether to use compilation to speed up token generation. Will increase startup time.
    """
    torch.set_float32_matmul_precision("high")

    checkpoint_dir = Path.home() / "checkpoints" / model_name

    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None and quantize in ["nf4", "nf4-dq"]:
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)

    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_json(checkpoint_dir / "lit_config.json")

    checkpoint_path = checkpoint_dir / "lit_model.pth" if checkpoint is None else checkpoint

    fabric.print(f"Loading model {str(checkpoint_path)!r}", file=sys.stderr)
    with fabric.init_module(empty_init=True):
        model = GPT(config)
        model.set_kv_cache(batch_size=1)
    load_checkpoint(fabric, model, checkpoint_path)
    model.eval()

    if compile:
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.coordinate_descent_tuning = True
        global next_token
        next_token = torch.compile(next_token, mode="reduce-overhead", dynamic=True)

    model = fabric.setup_module(model)

    tokenizer = Tokenizer(checkpoint_dir)

    system_prompt, stop_tokens = prompt_config(checkpoint_dir, tokenizer)

    L.seed_everything(1234)

    return dict(
        fabric=fabric,
        model=model,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        stop_tokens=stop_tokens,
        temperature=temperature,
        top_k=top_k
    )


def chat(*, fabric, model, tokenizer, system_prompt, stop_tokens, temperature, top_k, log_toks=True, prompt="", context=[], usage={}):
    prompt = system_prompt(prompt=prompt, context=context)
    encoded_prompt = tokenizer.encode(prompt, device=fabric.device)
    n_prompt_tokens = len(encoded_prompt)
    usage["prompt_tokens"] = n_prompt_tokens
    y = generate(
        model, encoded_prompt, model.max_seq_length, temperature=temperature, top_k=top_k, stop_tokens=stop_tokens
    )
    t0 = time.perf_counter()
    count = 0
    decoded = decode(fabric, tokenizer, y)
    context.append({
        "role": "user",
        "content": prompt
    })
    context.append({
        "role": "assistant",
        "content": ""
    })

    for token in decoded:
        count += 1
        usage["total_tokens"] = n_prompt_tokens + count
        context[-1]["content"] += token
        yield token
    t = time.perf_counter() - t0
    
    for block in model.transformer.h:
        block.attn.kv_cache.reset_parameters()

    if log_toks:
        fabric.print(
            f"\n\nTime for inference: {t:.02f} sec total, {count / t:.02f} tokens/sec, {count} tokens", file=sys.stderr
        )


if __name__ == "__main__":
    chat_config = setup_chat(model_name="microsoft/phi-1_5")
    prompt = " ".join(sys.argv[1:])
    chat(**chat_config, prompt=prompt)
