"""Final held-out metrics and qualitative examples, without training or tuning."""
import argparse
import json
import math
import time

import torch

from chat import generate
from prepare import ARTIFACTS, load_prepared, normalize_question
from utils import device_for, load_checkpoint, measure_loss, synchronize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--examples", type=int, default=6, help="Evenly spaced families for qualitative inspection")
    args = parser.parse_args()
    if args.examples < 1:
        parser.error("Examples must be positive.")
    torch.set_num_threads(4)
    device = device_for(args.device)
    model, tokenizer, checkpoint = load_checkpoint(device)
    data = load_prepared()
    if checkpoint["training"]["data_hash"] != data["data_hash"] or checkpoint["tokenizer"] != data["tokenizer"]:
        raise ValueError("Checkpoint and prepared data differ. Prepare and retrain together.")
    if checkpoint["training"]["smoke"]:
        raise ValueError("Final evaluation requires an actual training run, not a smoke checkpoint.")
    examples = data["splits"][args.split]
    loss, tokens = measure_loss(model, examples, 16, device)
    answer_loss, answer_tokens = measure_loss(model, examples, 16, device, answer_only=True)
    try:
        perplexity = math.exp(loss)
    except OverflowError:
        perplexity = None
    rows = [row for row in data["records"] if row["split"] == args.split]
    families = list(dict.fromkeys(row["family"] for row in rows))
    count = min(args.examples, len(families))
    selected_families = [families[i * (len(families) - 1) // max(1, count - 1)] for i in range(count)]
    selected = [next(row for row in rows if row["family"] == family) for family in selected_families]
    generate(model, tokenizer, selected[0]["question"], temperature=0, max_new_tokens=32)
    synchronize(device)
    started = time.perf_counter()
    outputs = [generate(model, tokenizer, row["question"], temperature=0) for row in selected]
    synchronize(device)
    duration = time.perf_counter() - started
    generated = sum(out["generated_tokens"] for out in outputs)
    chars = sum(len(normalize_question(r["question"])) + len(r["answer"]) for r in rows)
    unknown = sum(row.count(1) for row in examples)
    curves = checkpoint["training"]["curves"]
    first, last = curves[0], curves[-1]
    divergence = last["train_loss"] < first["train_loss"] and last["validation_answer_loss"] > min(c["validation_answer_loss"] for c in curves) + 0.05
    report = {"split": args.split, "objective": "all nonpadding next-token targets", "loss": loss,
              "protocol": "revision-v2: old test promoted to validation; fresh test frozen before development runs",
              "perplexity": perplexity, "perplexity_overflow": perplexity is None,
              "answer_only": {"loss": answer_loss, "target_tokens": answer_tokens,
                              "perplexity": math.exp(answer_loss) if answer_loss < 709 else None,
                              "objective": "answer characters and EOS only"},
              "examples": len(examples), "target_tokens": tokens, "unknown_characters": unknown,
              "character_count": chars, "unknown_character_fraction": unknown / chars,
              "vocabulary_size": len(tokenizer.labels), "parameter_count": model.parameter_count(),
              "checkpoint_bytes": (ARTIFACTS / "model.pt").stat().st_size,
              "data_hash": data["data_hash"], "training": checkpoint["training"],
              "generation": {"device": str(device), "warmup_generations": 1, "warmup_cap": 32,
                             "generation_count": len(outputs), "prompt_token_lengths": [o["prompt_tokens"] for o in outputs],
                             "generated_tokens_including_eos": generated, "seconds": duration,
                             "tokens_per_second": generated / duration, "temperature": 0, "max_new_tokens": 160,
                             "timing_scope": "encoding, model forward passes, sampling and decoding; no printing or checkpoint load; CUDA synchronized"},
              "qualitative_examples": [{"question": row["question"], "reference": row["answer"],
                                        "generated": out["answer"], "stop_reason": out["stop_reason"]}
                                       for row, out in zip(selected, outputs)],
              "curves_path": "artifacts/curves.json", "validation_divergence_flag": divergence,
              "curve_note": "Train loss includes dropout during updates; validation is evaluated without dropout."}
    path = ARTIFACTS / ("evaluation.json" if args.split == "test" else "validation-evaluation.json")
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"{args.split}: {len(examples)} examples, {tokens:,} target tokens; all-token loss {loss:.4f}; perplexity {perplexity}")
    print(f"Answer-only loss {answer_loss:.4f}; {answer_tokens:,} answer/EOS targets; perplexity {report['answer_only']['perplexity']}")
    print(f"Unknown characters {unknown}/{chars}; parameters {model.parameter_count():,}; checkpoint {report['checkpoint_bytes']:,} bytes")
    print(f"Greedy generation: {generated / duration:.1f} tokens/s on {device}; {len(outputs)} generations after one warmup")
    for example in report["qualitative_examples"]:
        print(f"\nQuestion: {example['question']}\nGenerated: {example['generated']}\nReference: {example['reference']}")
    print(f"\nValidation divergence flag: {divergence}; report: {path.name}; curves: artifacts/curves.json")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))
