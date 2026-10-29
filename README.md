# A tiny language model, built from scratch

This is an educational Python/PyTorch decoder-only transformer, trained from random initialization on a **fictional campus FAQ**. It demonstrates how a language model learns and generates text. It is a narrow demonstration, not a reliable campus assistant: generated answers can still contain wrong facts or fail on unfamiliar phrasing. No real college policy, personal record, external API, pretrained weights, retrieval system, or canned answer selector is involved.

## Run it in Windows PowerShell

Use Python 3.11 or another Python version supported by the selected PyTorch release. From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
python prepare.py
python train.py
python evaluate.py
python chat.py
python chat.py --temperature 0.7 --top-k 20
python chat.py --explain
python train.py --device cpu
python -m unittest discover -s tests -v
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python`; activation is optional. The local `.venv` already contains the CPU package used for the recorded run. Only PyTorch is a direct dependency; its installer supplies its own dependencies. This runtime emits a warning that optional NumPy support is missing. The project does not use NumPy, and the tests and full workflow pass without it.

The CPU index follows the [official PyTorch installation guidance](https://pytorch.org/get-started/locally/). For CUDA, choose your platform and compatible CUDA build in that official selector instead of installing a GPU package speculatively. `--device auto` uses CUDA when PyTorch reports it available, otherwise CPU. `--device cuda` fails clearly if unavailable. An RTX 4070 Laptop GPU with 8 GB VRAM was available for these runs, but these small runs used the existing CPU environment. No CUDA package was downloaded and no GPU training was performed.

Every resource path is relative to the Python source files, so invoking `python C:\path\to\Chatbot\chat.py` from another directory also works. Generated files go in `artifacts/`, which is ignored along with `.venv` and Python caches. Chat inputs are not logged or used to update weights.

For a quick pipeline check, run `python train.py --smoke --device cpu`. It uses at most 32 training and 16 validation records for two epochs and writes separate `smoke-model.pt` and `smoke-curves.json` files. It does not replace the real checkpoint or count as a completed training run. Training is capped at 150 epochs and 240 seconds by default; the time limit is checked after each epoch. On a slower machine it can stop before the learning period is complete, which is recorded in metadata and may leave a poor model. Controls include `--epochs`, `--max-seconds`, `--batch-size`, `--learning-rate`, `--seed`, `--device`, `--threads`, `--width`, `--objective`, `--selection-start`, and `--patience`.

## Files and workflow

| File | Purpose |
| --- | --- |
| `prepare.py` | Validate fixed splits, build a training-only tokenizer, encode independent examples |
| `model.py` | Explicit attention, transformer blocks, logits and next-token loss |
| `train.py` | AdamW updates, token-weighted losses, early stopping and checkpoint selection |
| `chat.py` | Autoregressive sampling, terminal loop and attention inspection |
| `evaluate.py` | Final held-out loss, perplexity, throughput and qualitative examples |
| `utils.py` | Shared device, checkpoint and loss measurement helpers |
| `data/conversations.jsonl` | Fixed, human-readable fictional corpus |
| `tests/test_lm.py` | Synthetic-fixture correctness tests |

Preparation writes `artifacts/tokenizer.json` and `artifacts/prepared.json`. Training writes the selected `artifacts/model.pt`, `artifacts/curves.json`, and a separate `last-model.pt` for development comparisons. Evaluation writes `artifacts/evaluation.json`. Use `python evaluate.py --split validation --examples 20` during development; it writes `validation-evaluation.json` without touching the final holdout. Checkpoints contain a state dict and primitive metadata, not a serialized model object. Loading uses `torch.load(..., weights_only=True)`, validates format/token IDs/dimensions/normalization, and loads the state dict strictly onto the chosen device. Evaluation rejects a checkpoint/data hash or tokenizer mismatch. Version 3 chat intentionally rejects older checkpoints because normalization changed. An explicit training-only `--init-from` supports continuing this project's compatible weights, including version 2, after checking identical character/token IDs; it never silently remaps vocabulary or downloads weights.

## Fictional fact sheet and data discipline

All facts below were invented for this project. Credits are made-up units, and dates do not describe any real institution.

| Topic | Fixed demo facts |
| --- | --- |
| Library | Cedar Hall beside the north entrance. Monday–Friday 8 am–8 pm; Saturday 10 am–4 pm; Sunday closed. Campus card required to borrow books for 14 days. First floor for groups; second floor for quiet study. |
| Parking | Students use the east blue lot with a 20-credit term permit requested through the portal. Visitors use the green lot beside the welcome desk, free for two hours. No overnight parking. Free bike racks outside Cedar Hall. |
| Wi-Fi | Students choose DemoNet and use student ID plus portal password. Guests choose DemoGuest and obtain a daily code from the welcome desk. First troubleshooting step: forget the network and reconnect. |
| Help desk | Maple Hall, weekdays 9 am–5 pm. Bring a campus card for recovery when the registered email is inaccessible. Never share passwords with the chatbot. |
| Registration | Courses section of the portal; August 1–20 inclusive. Ask an academic advisor about choices/prerequisites. Portal waitlists do not guarantee seats. Portal drops through September 5; registrar for later changes or a missed registration window. |
| Password reset | Forgot password on the portal sign-in page; link sent to registered email, including possible spam delivery. Links expire after 30 minutes; request another if expired. |
| Tuition | 600 credits per term, due September 1. Billing section of the portal. Two equal 300-credit installments can be arranged with the Birch Hall finance desk, which also explains aid. Tuition plus a parking permit is 620 credits. |
| Scope | Greetings and goodbyes; no real policies, medical advice, live weather, private records, dining menus, transport schedules, job listings, or ability to perform account actions. |

Each JSONL record explicitly declares its split and semantic paraphrase family:

```json
{"split":"train","family":"library_weekday","question":"What are the weekday library hours?","answer":"The demo library opens at 8 am and closes at 8 pm, Monday through Friday."}
```

There are **390 pairs, 90 families, and 6,875 whitespace-delimited question/answer words**. Concise answers were preferred over stretching toward 10,000–20,000 words. The split is 300 training examples (60 families), 60 validation (20 families), and 30 fresh test examples (10 families), approximately 77%/15%/8%. Every topic occurs across splits, but whole paraphrase families stay together. Held-out families ask distinct scenario or combined-fact questions. Consequently, the holdout is harder and generally longer than the direct training questions; loss differences reflect both distribution shift and overfitting.

`prepare.py` rejects duplicate case/punctuation/whitespace-normalized pairs, families spanning splits, missing/invalid fields, empty splits, and serialized records longer than 257 tokens. The longest unaugmented records are 132/175/171 tokens in train/validation/test. The closest cross-split questions were ranked with the standard-library `difflib.SequenceMatcher` and manually inspected before final evaluation: the highest character similarity was about 0.659, with different questions about student parking and transport. This is a review aid, not a proof that all semantic overlap is absent. Shared fictional facts are intentional.

Only training text builds the vocabulary. Only validation guides checkpoint selection and development. The original test set was used in the earlier quality discussion, so all its 30 examples were explicitly promoted to validation, never training. A new 30-example holdout was authored and frozen before the revision experiments. Final holdout outputs must not be used to change model settings or retrain without reserving another fresh holdout. Unsupported-request examples teach some uncertainty phrases, but do not guarantee reliable refusal or factuality. This evaluates next-character prediction in a small synthetic domain, not general conversation or reliable knowledge.

## From characters to learned weights

The tokenizer reserves six integer IDs: PAD=0, UNK=1, BOS=2, USER=3, ASSISTANT=4, EOS=5. Questions are lowercased, whitespace is collapsed, and trailing spaces/periods/question marks/exclamation marks are replaced with one `?`, identically during preparation and chat. Thus `What are the library hours?` and `what are the library Hours` have identical model inputs. Answer case and punctuation are preserved. The underlying character encoder does not silently normalize arbitrary text. Sorted characters seen in **normalized training questions and training answers only** receive IDs starting at 6. Ordinary spaces and punctuation are characters. Special tokens are inserted as single IDs; their spelled-out labels are not encoded letter by letter. The vocabulary has 68 tokens. Unseen characters map to UNK without extending or reordering IDs. Decoding skips structural tokens and renders UNK as `�`; generation forbids UNK as well as PAD/BOS/USER/ASSISTANT.

An example is serialized as `[BOS, USER] + question_characters + [ASSISTANT] + answer_characters + [EOS]`. For every record independently, `x = tokens[:-1]` and `y = tokens[1:]`. Batches are right-padded to their longest input. There is no concatenation across examples. A 257-token serialized example provides at most 256 prediction positions.

```text
IDs [B,T]
   -> token embeddings + learned absolute position embeddings [B,T,128]
   -> 2 pre-normalized transformer blocks
        LayerNorm -> Q/K/V -> 4-head causal attention -> dropout -> residual add
        LayerNorm -> Linear(128,512) -> GELU -> Linear(512,128) -> dropout -> residual add
   -> final LayerNorm
   -> vocabulary projection tied to token embeddings
   -> logits [B,T,68]
   -> shifted next-token cross-entropy -> backpropagation -> AdamW updates
```

Each token embedding is a learned vector. Position embeddings distinguish an ordering such as “book a room” from rearrangements of the same characters. In each attention head, linear projections produce queries, keys and values with head width 32. Queries compare with keys using `Q @ K.transpose(-2, -1) / sqrt(32)`, giving scores `[B,4,T,T]`. Scaling keeps dot products from growing excessively with head width and saturating softmax. A registered lower-triangular mask hides future keys, and a padding mask hides padded keys. Softmax normalizes allowed scores into weights that mix value vectors. The four heads are concatenated and projected back to width 128.

Every valid query has its own valid key. For a completely masked padded query, the implementation allows a harmless self edge to keep softmax finite; padded targets never enter the loss. Residual additions preserve a direct path for information and gradients. The feed-forward network transforms each position independently. LayerNorm stabilizes activations. Dropout is 0.1 during training and disabled during evaluation.

The final projection shares its matrix with the token embeddings, so parameter counting includes it once. Linear/embedding matrices start with normal standard deviation 0.02; linear biases start at zero and LayerNorm uses its usual defaults. The training CLI now uses width 128 and **438,272 unique parameters**, below the enforced 500,000-parameter limit. `Config` retains a width-96 default for small fixtures and explicit baseline comparisons; checkpoints record their actual dimensions.

Logits are unnormalized scores. Cross-entropy flattens them to `[B*T,V]` and targets to `[B*T]`, then rewards higher probability for the actual next token. The revised default is **answer-focused next-token training**: question and structural-marker targets are ignored, while every answer character and its final EOS contribute to loss. The question remains visible as causal context, and gradients can flow through its embeddings and attention. `--objective all` retains the original standard language-model objective over every nonpadding target. Evaluation always reports full-sequence loss/perplexity and separately labeled answer-only loss/perplexity with the correct token count. Full-sequence perplexity may get worse under answer-focused training; it must not be presented as an improvement in general language modeling.

Backpropagation computes derivatives of the scalar loss with respect to parameters. AdamW updates weights using gradients and weight decay; gradient norms are clipped to 1.0. Losses are weighted by the count of included target tokens, not by batches. During training, 70% of questions are eligible for neutral prefixes/suffixes such as “please help with this question.” This varies length without changing an answer, split, or fact. Overlong augmented examples fall back to their original form without dropping targets; `--no-question-augmentation` disables it. The experimental `--position-jitter` shifts learned position IDs during training; it was used in development but is off by default and was disabled for the saved model's refinement pass because repeated-letter errors remained. Evaluation positions always start at zero. Augmented variants are not counted as additional authored records or independent evaluation examples.

Training sees the full known sequence and computes all next-token positions in parallel. Causal masking makes each position unable to see its own future target. Generation does not know future characters, so it must append one sampled token and run the model again. It is a language model because it predicts a distribution over next tokens and forms new sequences autoregressively; characters are simply a smaller token unit than the subwords commonly used by larger models.

Trainable **parameters** include embeddings, Q/K/V matrices, MLP weights and normalization scales. **Activations** are temporary representations produced for a particular input. **Attention weights** are input-dependent softmax values computed from Q and K, not a separately stored table of learned answers. The checkpoint saves parameters; ordinary training does not retain attention histories.

## Generation and inspection

`chat.py` loads the best saved weights, formats `[BOS, USER] + question + [ASSISTANT]`, takes the final-position logits, samples one token, appends it, and repeats. Only generated answer characters are printed. EOS ends the answer; otherwise the default cap is 160 new tokens, including EOS if generated. Empty input skips the model. `/help`, `/quit`, EOF and Ctrl+C are supported. Questions are independent, with no conversation memory or continual learning.

Temperature 0 is now the default and uses greedy argmax for repeatable demonstrations. Positive temperatures divide eligible logits before softmax: lower values concentrate probability and higher values spread it. Try `--temperature 0.5 --top-k 20 --seed 42` to demonstrate sampling, which can introduce more mistakes. Top-k keeps only the k highest eligible logits; 0 disables filtering and oversized values are clamped to the eligible vocabulary size. PAD, UNK, BOS, USER and ASSISTANT cannot be sampled; EOS can. There is no repetition penalty or canned fallback to hide failures.

A question longer than 253 characters is truncated at its end to preserve all three prompt markers, with a visible notice. As prompt plus generation exceeds 256 tokens, the model sees only the most recent 256 and absolute positions restart at zero. This simple sliding window can lose the original role markers and earlier question context. There is no KV cache.

`--explain` reports dimensions, parameter count and device; the first next-token top-five eligible probabilities at temperature 1 before top-k; and strongest last-block attention positions from the final prompt token, averaged across heads before dropout. Position numbers keep repeated characters distinguishable. It also reports sampling settings, token count and EOS/length-cap stop reason. These token probabilities are **not answer confidence**. Attention strengths do not prove causal importance or correctness.

## Current saved progress

The saved model generates readable answers to several short, familiar campus questions. It is still **unreliable on longer or unfamiliar questions**, often producing fluent but unrelated answers, and sometimes misspelling words. The new holdout results are poor. This is a partial improvement in familiar-prompt readability, not a successful general-purpose chatbot or evidence of improved generalization.

The saved weights came from a 150-epoch, width-128 answer-focused run with question augmentation and position offsets, followed by 30 refinement epochs at learning rate 0.0003 with position offsets disabled and canonical terminal punctuation. All weights originated in this project's random initialization; no external pretrained weights were used. The base run took 211.21 seconds and refinement took 43.00 seconds on CPU. Across the five development runs, measured training time was about 488 seconds, excluding setup, tests and inference. GPU training was not used.

Checkpoint selection changed during development based on validation behavior. Early minimum-loss checkpoints still produced gibberish. The longer run was compared qualitatively on development/validation prompts, and refinement used a fixed 30-epoch budget (`--selection-start 30`). Therefore the current model is **not the global minimum-validation-loss model**. Metadata retains the initial run and refinement history; `best_epoch` refers to the best within the configured selection window, not all historical epochs. Low held-out loss and coherent factual answers did not improve together in these experiments.

| Current measurement | Value |
| --- | --- |
| Unique trainable parameters | 438,272 |
| Vocabulary | 68 tokens |
| Fresh test examples / full-sequence target tokens | 30 / 3,853 |
| Full-sequence test loss / perplexity | 5.2057 / 182.3084 |
| Answer-only test loss / perplexity | 4.4736 / 87.6761 |
| Answer/EOS target tokens | 1,764 |
| Unknown characters / normalized character count | 7 / 3,763 |
| Checkpoint bytes | 1,909,785 |
| Measured CPU generation speed | 1,080.7 tokens/second |

Full-sequence metrics include question and structural targets, even though training focused on answer characters and EOS. Perplexity is not accuracy; the two reported objectives must not be confused. The old baseline used a different holdout and training objective, so its 13.47 perplexity is not a like-for-like comparison. The fresh holdout was evaluated once after the model was frozen. Its outputs did not inform subsequent training. If the next iteration uses these observed failures for tuning, treat this holdout as validation and reserve another fresh final set.

Throughput measured 543 generated tokens including EOS across ten fixed family-spanning prompts after one warmup capped at 32 tokens. Timing was 0.502439 seconds and included encoding, model calls, selection and decoding, but excluded printing and checkpoint loading. Generation was greedy, capped at 160 new tokens; prompt lengths are recorded in `artifacts/evaluation.json`. It is a short local measurement, not a hardware benchmark. CUDA timers synchronize when applicable, but CUDA was not exercised.

Verbatim development output:

> **Question:** What are the library hours?
>
> **Generated:** The library opens at 8 am and closes at 8 pm on weekdays.

Verbatim fresh-holdout failure:

> **Question:** My visitor needs parking for three hours. Is the whole stay free?
>
> **Generated:** Reset links expire after 30 minutes. Request a new link if needed.
>
> **Reference:** Visitor parking is free for two hours in the green lot. I do not know the fee after that.

The reference and generated answer are unrelated. The full report includes ten fixed qualitative examples rather than a selected fluent subset. The refinement training loss fell from 0.0704 to 0.0294 while validation answer loss worsened from 3.0791 to 3.2052, consistent with overfitting. There are no claims of calibrated confidence, reliable factual knowledge or reasoning ability.

The old baseline files are preserved in `artifacts/baseline-v1/`. Development candidates are in `artifacts/revision-all/`, `artifacts/revision-jitter-all/`, `artifacts/revision-answer/`, and `artifacts/revision-augmented/`. The latter contains the pre-refinement `last-model.pt`. To repeat the recorded refinement locally:

```powershell
.\.venv\Scripts\python.exe train.py --device cpu --init-from artifacts/revision-augmented/last-model.pt --epochs 30 --selection-start 30 --learning-rate 0.0003 --max-seconds 60
```

This is a second phase starting from locally trained weights, not a fresh run. A plain `python train.py` starts from random weights with the currently documented defaults and is not guaranteed to recreate the two-phase checkpoint. All generated artifacts are ignored by Git; the local checkpoint is present, but distributing source alone does not include it.

## Verification and interview discussion

All **19 CPU unit tests passed**, including tokenizer stability and unknown handling; training-only vocabulary and leakage constraints; overlength rejection; shifted targets and ignored padding; finite forward passes; future-token isolation; attention normalization and masking; padding/batch invariance; nonzero finite gradients through embeddings and Q/K/V; a deterministic 120-update synthetic overfit check; checkpoint/logit round-trip; parameter budget; greedy determinism, sampling restrictions, EOS and length caps; terminal empty/help/quit/EOF/Ctrl+C handling; answer-mask alignment and weighting; question normalization; position-jitter bounds; and augmentation target preservation. Test fixtures do not use the final holdout to set thresholds.

The revised two-epoch smoke pass, bounded training/refinement runs, final held-out evaluation and scripted chat checks completed. The terminal checks covered a known-style prompt, an unfamiliar prompt, explain mode, truncation, empty input, help, quit and EOF, including invocation from another working directory. Ctrl+C was checked by injecting `KeyboardInterrupt` into the input loop. Missing-checkpoint instructions, invalid temperature and unavailable-CUDA errors were also checked. CUDA hardware execution and a physical interactive Ctrl+C keypress were not tested.

A future extension could manually revise the corpus and compare changes using validation. If final test outputs guide revisions, that set has effectively become validation and a fresh final holdout is needed. There is no feedback database, and typing into the chatbot does not train it.
