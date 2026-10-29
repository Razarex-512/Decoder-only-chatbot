"""Train from random initialization; select weights using validation only."""
import argparse
import json
import hashlib
from pathlib import Path
import random
import time

import torch

from model import Config, TinyLM, next_token_loss
from prepare import ARTIFACTS, Tokenizer, answer_targets, batchify, load_prepared
from utils import device_for, load_checkpoint, measure_loss, save_checkpoint, seed_all, synchronize


def training_examples(records, tokenizer, augment):
    """Vary question length/style without changing facts, answers or split membership."""
    prefixes = ["", "hello, ", "i am new to this campus. ", "please help with this question: ",
                "i would like some campus information. ", "i am planning a visit and need some help. "]
    suffixes = ["", " please.", " could you explain?", " i would appreciate your help.",
                " this would help me plan my day."]
    examples = []
    for row in records:
        question = row["question"]
        if augment and random.random() < 0.7:
            question = random.choice(prefixes) + question + random.choice(suffixes)
        encoded = tokenizer.record(question, row["answer"])
        # Never truncate or drop a target to fit augmentation.
        if len(encoded) > 257:
            encoded = tokenizer.record(row["question"], row["answer"])
        examples.append(encoded)
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--objective", choices=["all", "answer"], default="answer")
    parser.add_argument("--patience", type=int, default=75)
    parser.add_argument("--selection-start", type=int, default=75, help="Start checkpoint selection after this learning period")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--max-seconds", type=float, default=240, help="Training wall-time budget, checked after each epoch")
    parser.add_argument("--position-jitter", action="store_true", help="Enable experimental training-time random position offsets")
    parser.add_argument("--no-question-augmentation", action="store_true")
    parser.add_argument("--init-from", type=Path, help="Optional continuation from this project's own compatible checkpoint")
    parser.add_argument("--threads", type=int, default=4, help="CPU worker threads; small models benefit from few threads")
    parser.add_argument("--smoke", action="store_true", help="Two epochs on <=32 train/16 validation records; separate checkpoint")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 150 or min(args.batch_size, args.threads) < 1 or not 0 < args.learning_rate < float("inf"):
        parser.error("Use 1..150 epochs, positive batch size/threads, and finite positive learning rate.")
    if min(args.patience, args.selection_start, args.width) < 1 or args.width % 4 or not 0 < args.max_seconds < float("inf"):
        parser.error("Patience, selection start, width and time budget must be positive; width must divide into 4 heads.")
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    device = device_for(args.device)
    data = load_prepared()
    tokenizer = Tokenizer.from_metadata(data["tokenizer"])
    model = TinyLM(Config(len(tokenizer.labels), width=args.width, position_jitter=args.position_jitter)).to(device)
    initialization = None
    if args.init_from:
        model, previous_tokenizer, previous = load_checkpoint(device, args.init_from, allow_previous=True)
        if previous_tokenizer.characters != tokenizer.characters:
            raise ValueError("Continuation vocabulary differs. No token IDs may be silently remapped; train from scratch.")
        model.config.position_jitter = args.position_jitter
        initialization = {"checkpoint_sha256": hashlib.sha256(args.init_from.read_bytes()).hexdigest(),
                          "source_format": previous["format_version"], "source_training": previous["training"],
                          "source_is_last_epoch": args.init_from.name == "last-model.pt"}
    count = model.parameter_count()
    print(f"Unique trainable parameters: {count:,}; device: {device}", flush=True)
    if count >= 500_000:
        raise ValueError("Model exceeds the parameter budget.")
    validation = data["splits"]["validation"]
    train_records = [r for r in data["records"] if r["split"] == "train"]
    if args.smoke:
        validation = validation[:16]
        train_records = train_records[:32]
    epochs = min(args.epochs, 2) if args.smoke else args.epochs
    selection_start = 1 if args.smoke else min(args.selection_start, epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    path = ARTIFACTS / ("smoke-model.pt" if args.smoke else "model.pt")
    curve_path = ARTIFACTS / ("smoke-curves.json" if args.smoke else "curves.json")
    metadata = {"seed": args.seed, "device": str(device), "torch_version": str(torch.__version__),
                "parameter_count": count, "data_hash": data["data_hash"], "batch_size": args.batch_size,
                "optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": 0.01,
                "gradient_clip": 1.0, "patience": args.patience, "maximum_epochs": epochs, "threads": args.threads,
                "max_seconds": args.max_seconds, "smoke": args.smoke,
                "position_jitter": args.position_jitter, "initialization": initialization,
                "question_augmentation": not args.no_question_augmentation, "selection_start_epoch": selection_start,
                "objective": "all nonpadding next-token targets" if args.objective == "all" else "answer characters and EOS only",
                "selection_metric": "validation answer-only token-average loss after learning period", "curves": []}
    best, stale = float("inf"), 0
    synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = training_examples(train_records, tokenizer, not args.no_question_augmentation)
        random.shuffle(order)
        total, tokens = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            x, y = batchify(order[start:start + args.batch_size], device)
            if args.objective == "answer":
                y = answer_targets(x, y)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = next_token_loss(logits, y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; no checkpoint selected from this update.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = y.ne(0).sum().item()
            total += loss.item() * n
            tokens += n
        val_loss, _ = measure_loss(model, validation, args.batch_size, device)
        answer_loss, _ = measure_loss(model, validation, args.batch_size, device, answer_only=True)
        synchronize(device)
        elapsed = time.perf_counter() - started
        metadata["curves"].append({"epoch": epoch, "train_loss": total / tokens,
                                    "validation_loss": val_loss, "validation_answer_loss": answer_loss,
                                    "elapsed_seconds": elapsed})
        print(f"epoch {epoch:3d} | train {total / tokens:.4f} | val-all {val_loss:.4f} | val-answer {answer_loss:.4f} | {elapsed:.1f}s", flush=True)
        if epoch == selection_start:
            best, stale = float("inf"), 0
        if answer_loss < best:
            best, stale = answer_loss, 0
            metadata.update(best_epoch=epoch, best_validation_loss=val_loss, best_validation_answer_loss=best)
            save_checkpoint(path, model, tokenizer, metadata)
        else:
            stale += 1
        if (epoch >= selection_start and stale >= args.patience) or elapsed >= args.max_seconds:
            break
    # Preserve the final weights for transparent development comparisons; chat
    # still uses only the checkpoint selected by validation, never this file.
    if not args.smoke:
        save_checkpoint(ARTIFACTS / "last-model.pt", model, tokenizer, metadata)
    model, tokenizer, _ = load_checkpoint(device, path)
    metadata["duration_seconds"] = time.perf_counter() - started
    metadata["epochs_completed"] = epoch
    metadata["learning_period_completed"] = epoch >= selection_start
    metadata["stop_reason"] = "time budget" if elapsed >= args.max_seconds else "early stopping" if stale >= args.patience else "epoch limit"
    save_checkpoint(path, model, tokenizer, metadata)  # Best weights plus complete run history.
    curve_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    final_loss, _ = measure_loss(model, validation, args.batch_size, device)
    final_answer_loss, _ = measure_loss(model, validation, args.batch_size, device, answer_only=True)
    print(f"Reloaded selected epoch {metadata['best_epoch']}; validation all {final_loss:.4f}, answer {final_answer_loss:.4f}; saved {path.name}")
    if not metadata["learning_period_completed"]:
        print("Time budget ended before the learning period completed; this checkpoint may still be incoherent.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))
