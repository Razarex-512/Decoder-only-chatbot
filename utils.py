"""Shared device, measurement and safe checkpoint helpers."""
from dataclasses import asdict
import random
import torch

from model import Config, TinyLM, next_token_loss
from prepare import ARTIFACTS, FORMAT_VERSION, SPECIAL, Tokenizer, answer_targets, batchify


def device_for(request="auto"):
    if request == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu.")
    return torch.device("cuda" if request == "auto" and torch.cuda.is_available()
                        else "cpu" if request == "auto" else request)


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def measure_loss(model, examples, batch_size, device, answer_only=False):
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(examples), batch_size):
        x, y = batchify(examples[start:start + batch_size], device)
        logits, _ = model(x)
        if answer_only:
            y = answer_targets(x, y)
        n = y.ne(0).sum().item()
        total += next_token_loss(logits, y).item() * n
        count += n
    return total / count, count


def save_checkpoint(path, model, tokenizer, training):
    torch.save({"format_version": FORMAT_VERSION, "architecture": asdict(model.config),
                "tokenizer": tokenizer.metadata(), "training": training,
                "state_dict": model.state_dict()}, path)


def load_checkpoint(device, path=ARTIFACTS / "model.pt", allow_previous=False):
    if not path.exists():
        raise ValueError("Checkpoint missing. Run python prepare.py, then python train.py.")
    payload = torch.load(path, map_location=device, weights_only=True)
    previous = allow_previous and payload.get("format_version") == 2
    if payload.get("format_version") != FORMAT_VERSION and not previous:
        raise ValueError("Unsupported checkpoint format. Retrain with this project version.")
    if previous:
        # Explicit training-only transfer. No token IDs are remapped, and chat
        # never accepts the older normalization. Training checks every character.
        if payload["tokenizer"].get("special_ids") != SPECIAL or payload["tokenizer"].get("question_normalization") != "lower-collapse-whitespace-v1":
            raise ValueError("Unsupported tokenizer transfer.")
        tokenizer = Tokenizer(payload["tokenizer"]["characters"])
    else:
        tokenizer = Tokenizer.from_metadata(payload["tokenizer"])
    config = Config(**payload["architecture"])
    if config.vocab_size != len(tokenizer.labels) or config.pad_id != 0 or config.context > 256:
        raise ValueError("Checkpoint architecture and tokenizer are incompatible.")
    model = TinyLM(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, tokenizer, payload
