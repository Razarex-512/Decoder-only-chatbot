"""Autoregressive character generation and a deliberately simple terminal UI."""
import argparse
import math

import torch

from utils import device_for, load_checkpoint, seed_all
from prepare import normalize_question


def prompt_tokens(tokenizer, question, context):
    encoded = tokenizer.encode(normalize_question(question))
    # Retain the start of a long question and all three structural markers.
    return [2, 3] + encoded[:context - 3] + [4], len(encoded) > context - 3


def sampling_distribution(logits, temperature, top_k):
    if not math.isfinite(temperature) or temperature < 0 or top_k < 0:
        raise ValueError("Temperature must be finite and nonnegative; top-k must be nonnegative.")
    scores = logits.clone()
    scores[:5] = float("-inf")  # PAD, UNK, BOS, USER, ASSISTANT; EOS is allowed.
    if temperature == 0:
        return torch.nn.functional.one_hot(scores.argmax(), scores.numel()).to(scores.dtype)
    scores /= temperature
    eligible = scores.numel() - 5
    if top_k:
        indices = scores.topk(min(top_k, eligible)).indices
        filtered = torch.full_like(scores, float("-inf"))
        filtered[indices] = scores[indices]
        scores = filtered
    return scores.softmax(dim=-1)


@torch.inference_mode()
def generate(model, tokenizer, question, temperature=0.0, top_k=20, max_new_tokens=160, explain=False):
    if max_new_tokens < 1:
        raise ValueError("Maximum new tokens must be positive.")
    # Validate even if a fixture would immediately emit EOS.
    if not math.isfinite(temperature) or temperature < 0 or top_k < 0:
        raise ValueError("Temperature must be finite and nonnegative; top-k must be nonnegative.")
    model.eval()
    device = next(model.parameters()).device
    prompt, truncated = prompt_tokens(tokenizer, question, model.config.context)
    history, generated, diagnostic = list(prompt), [], {}
    reason = "length cap"
    for step in range(max_new_tokens):
        x = torch.tensor([history[-model.config.context:]], dtype=torch.long, device=device)
        logits, attention = model(x, inspect=explain and step == 0)
        if explain and step == 0:
            # Unfiltered eligible next-token distribution, T=1; not answer confidence.
            probabilities = sampling_distribution(logits[0, -1], 1.0, 0)
            values, indices = probabilities.topk(min(5, len(tokenizer.labels) - 5))
            diagnostic["next_tokens"] = [(tokenizer.labels[i], p) for i, p in zip(indices.tolist(), values.tolist())]
            weights = attention[-1][0, :, -1, :].mean(dim=0)
            values, positions = weights.topk(min(6, len(prompt)))
            diagnostic["attention"] = [(i, tokenizer.labels[prompt[i]], p)
                                       for i, p in zip(positions.tolist(), values.tolist())]
        probabilities = sampling_distribution(logits[0, -1], temperature, top_k)
        token = probabilities.argmax().item() if temperature == 0 else torch.multinomial(probabilities, 1).item()
        generated.append(token)
        history.append(token)
        if token == 5:
            reason = "EOS"
            break
    return {"answer": tokenizer.decode(generated), "token_ids": generated,
            "generated_tokens": len(generated), "stop_reason": reason, "truncated": truncated,
            "prompt_tokens": len(prompt), "diagnostic": diagnostic}


def print_explanation(model, result, args, device):
    c = model.config
    print(f"  Parameters {model.parameter_count():,}; width {c.width}; blocks {c.layers}; heads {c.heads}; context {c.context}; {device}")
    print("  First next-token probabilities (eligible tokens, temperature 1, before top-k; NOT answer confidence):")
    for label, probability in result["diagnostic"]["next_tokens"]:
        print(f"    {label!r}: {probability:.4f}")
    print("  Last-block attention from final prompt position, averaged over heads (position, token, weight):")
    for position, label, weight in result["diagnostic"]["attention"]:
        print(f"    {position}: {label!r} {weight:.4f}")
    print(f"  temperature={args.temperature}; top-k={args.top_k} (clamped to eligible vocabulary); generated={result['generated_tokens']}; stop={result['stop_reason']}")
    print("  Q/K/V matrices are parameters; attention is input-dependent and does not prove correctness or causal importance.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 uses repeatable greedy decoding (default); positive values sample")
    parser.add_argument("--top-k", type=int, default=20, help="0 disables filtering")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.temperature) or args.temperature < 0 or args.top_k < 0 or args.max_new_tokens < 1:
        parser.error("Use finite temperature >=0, top-k >=0, and max-new-tokens >=1.")
    torch.set_num_threads(4)
    if args.seed is not None:
        seed_all(args.seed)
    device = device_for(args.device)
    model, tokenizer, _ = load_checkpoint(device)
    print("Tiny educational language model. All campus content is fictional; answers may be nonsense.")
    print("Each question is independent. /help for commands; /quit to exit.")
    while True:
        try:
            question = input("You: ").strip()
            if not question:
                continue
            if question == "/quit":
                break
            if question == "/help":
                print("Ask about the fictional campus. /help shows this help; /quit exits. No memory or automatic learning.")
                continue
            result = generate(model, tokenizer, question, args.temperature, args.top_k, args.max_new_tokens, args.explain)
            if result["truncated"]:
                print("[Question truncated to fit context; role markers preserved.]")
            print(result["answer"])
            if args.explain:
                print_explanation(model, result, args, device)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))
