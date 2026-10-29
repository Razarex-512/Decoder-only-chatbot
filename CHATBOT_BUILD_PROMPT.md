# Build a Tiny Generative Language Model From Scratch

## Mission

Implement a complete educational Python project that trains a tiny decoder-only transformer from random initialization and uses it as a terminal chatbot. The user wants a demonstration of language-model fundamentals, not a production assistant or deployable LLM. Complete implementation and meaningful verification, rather than returning another plan.

Inspect applicable AGENTS.md instructions and existing files first. Preserve unrelated changes. Make reasonable small implementation decisions autonomously. Keep explanations readable for a student who wants to discuss every component in an interview.

## Non-negotiable scope

- Actually generate text autoregressively; do not replace generation with intent classification, answer retrieval, templates, or keyword routing.
- Implement token embeddings, position embeddings, causal multihead self-attention, feed-forward layers, residual connections, normalization, next-token loss, backpropagation, and sampling.
- Train all model weights from scratch. No pretrained checkpoints, hosted APIs, downloaded language models, Transformers library, LangChain, vector database, or external service.
- Python and PyTorch only; standard library for data processing, CLI, metrics, and tests.
- CPU must support the complete workflow. CUDA is optional, automatically selected when available, with an explicit CPU override.
- No frontend, web server, database, Docker, authentication, deployment, or unnecessary package hierarchy.
- Keep the model below 500,000 unique trainable parameters. Target roughly 150,000-300,000 and print the actual count.
- Aim for roughly 500-750 readable lines across core Python modules. Prefer clarity over artificially meeting a line count.
- Be candid: a tiny model on a tiny synthetic corpus may memorize, repeat, or produce nonsense. Demonstrating the mechanics successfully is the primary goal.
- Never invent measured accuracy, quality improvements, training time, GPU usage, or historical project dates.

## Expected experience

```powershell
python prepare.py
python train.py
python evaluate.py
python chat.py
python chat.py --temperature 0.7 --top-k 20
python chat.py --explain
python train.py --device cpu
python -m unittest discover -s tests
```

The chat loop shows that this is an educational model with fictional training content. Each question is independent; conversation memory is out of scope. It formats a user prompt, samples new tokens, and displays only the generated answer. Support `/help`, `/quit`, EOF, and Ctrl+C. Handle empty input without running the model. Limit generation length and avoid unbounded loops. Do not hide incoherent outputs behind canned answers.

## Project structure

```text
model.py                 # Transformer, attention, forward pass
prepare.py               # Dataset checks, tokenizer, encoded splits
train.py                 # Training, validation, checkpointing
chat.py                  # Sampling, CLI, explanation
evaluate.py              # Held-out loss, perplexity, generation examples
data/conversations.jsonl # Explicit fixed train/validation/test records
tests/test_lm.py         # Focused correctness tests
requirements.txt         # PyTorch only
README.md
.gitignore
artifacts/               # Generated tokenizer, encodings, checkpoints, results
```

Use these filenames without leading spaces. Share helpers sensibly; add one small utility module only if it avoids real duplication. Resolve resources relative to source-file locations so invocation works from another working directory. Ignore generated artifacts and Python caches. Do not commit local conversation text or machine-specific paths.

## Step 1: Author and split a small corpus

Use a fictional campus FAQ domain to constrain vocabulary and keep answers short. Cover greetings, library hours, parking, Wi-Fi, registration, password reset, tuition, and goodbyes. All policy details must be explicitly fictional. Do not copy personal information from the user's resume.

Author roughly 250-400 concise question/answer pairs, around 10,000-20,000 words total if practical. Quality matters more than hitting the word count. Use varied, human-readable phrasing rather than hundreds of near-identical substitutions. Keep answers consistent with a small written fictional fact sheet in the README. Include some unsupported requests paired with honest uncertainty, while explaining that these do not guarantee reliable rejection.

Represent each JSONL record as:

```json
{"split":"train","family":"library_weekday_hours","question":"When does the library close on weekdays?","answer":"The demo campus library closes at 8 pm on weekdays."}
```

- Assign approximately 80% train, 10% validation, 10% test.
- Assign entire paraphrase families to one split; do not randomly scatter sibling variants across splits.
- Ensure topics occur across splits using distinct families or question forms. Shared domain facts are permitted; duplicated or nearly identical question-answer pairs are not.
- Reject identical normalized pairs across splits and reject families spanning splits. Manually inspect for near duplicates.
- Keep explicit split membership fixed before training.
- Train data constructs the tokenizer. Validation selects checkpoints. Test measures the final chosen model only.
- Explain that this evaluates generalization within a small synthetic domain, not broad language ability or guaranteed factual knowledge.
- Corpus stays local and readable; no automated web scraping or large download.

## Step 2: Build a character tokenizer

Use character-level tokenization for transparency and minimal dependencies. This will generate text character by character; explain why that is still a language model and why it differs from modern subword tokenization.

Reserve dedicated integer token IDs for `<pad>`, `<unk>`, `<bos>`, `<user>`, `<assistant>`, and `<eos>`. These are special tokens inserted programmatically, not character strings to be learned letter by letter. Remaining IDs represent sorted characters observed in training only. Preserve ordinary case and punctuation consistently. Map unseen validation/test/input characters to `<unk>` without expanding vocabulary.

Serialize each example as:

```text
<BOS><USER>question characters<ASSISTANT>answer characters<EOS>
```

Document the exact encode/decode behavior, skip structural tokens when displaying generated text, and never silently remap token IDs at inference. Store the vocabulary and special-token IDs as JSON-compatible metadata.

Use a maximum context of 256 tokens. Keep authored serialized examples within 257 tokens where possible, so input/target shifting can retain up to 256 prediction positions. Explicitly reject overlong training records during preparation with actionable errors; do not silently drop them. For live prompts, truncate the question to fit while preserving BOS, USER, and ASSISTANT markers, and tell the user when truncation occurs.

## Step 3: Prepare next-token batches

For each encoded example, create:

```text
x = tokens[:-1]
y = tokens[1:]
```

Pad batches to their longest input. Loss ignores padded targets. Keep each example independent; do not concatenate examples in ways that leak cross-record context.

Default training objective: standard causal next-token cross-entropy over every nonpadding target, including question, answer, and structural tokens. Report it accurately. Optionally report answer-only loss as an additional diagnostic using a target-position mask, but do not complicate the first implementation with separate training objectives.

Document shapes: token IDs `[B,T]`, embeddings `[B,T,C]`, attention scores `[B,H,T,T]`, logits `[B,T,V]`, flattened cross-entropy inputs `[B*T,V]` and `[B*T]`.

## Step 4: Implement a tiny decoder-only transformer

Default architecture:

- Context length: 256.
- Embedding width: 96.
- Transformer blocks: 2.
- Attention heads: 4, each with head width 24.
- Feed-forward hidden width: 384 (4 times embedding width).
- Dropout: 0.1 during training.
- Learned token and absolute position embeddings.
- Pre-normalization blocks:

```text
a = x + dropout(attention(layer_norm_1(x)))
x = a + dropout(feed_forward(layer_norm_2(a)))
```

- Feed-forward: Linear -> GELU -> Linear.
- Final LayerNorm, then linear vocabulary projection.
- Tie vocabulary projection weights to token embedding weights to reduce parameters. Count unique parameters correctly.

Implement explicit Q/K/V projections and scaled dot-product attention so the mechanics remain visible:

```text
scores = Q @ K.transpose(-2, -1) / sqrt(head_dim)
scores = mask_future_and_padding_keys(scores)
weights = softmax(scores, dim=-1)
output = dropout(weights) @ V
```

Use a registered lower-triangular causal mask sliced to the active sequence length. Mask padded keys. Ensure each valid query has at least one valid key and avoid NaNs; padded query positions are ignored by loss. Concatenate heads, project back to embedding width, and apply the residual path.

Return logits and optional attention matrices for inspection. Avoid storing attention histories during ordinary training. Explain that causal masking prevents access to future target information. Do not add KV caching, rotary embeddings, flash attention, mixture-of-experts, or other optimizations unnecessary at this scale.

Initialize weights sensibly using a small normal distribution and normal LayerNorm defaults. Print the parameter count and enforce the 500,000 budget before training.

## Step 5: Train and save weights

Provide argparse controls for seed, device, epochs, batch size, learning rate, and a bounded smoke-test mode. Defaults:

- Seed 42 for Python and PyTorch; seed CUDA if available.
- Batch size 16.
- AdamW with learning rate 0.0003 and weight decay 0.01.
- Maximum 150 epochs.
- Gradient clipping norm 1.0.
- Early stopping patience 15 validations based on validation token-average loss.

Keep a constant learning rate initially; no scheduler is needed. Shuffle training records only. Use `train()` for updates and `eval()` plus inference/no-grad mode for validation. Compute epoch loss weighted by the number of nonignored target tokens, not an unweighted average of batch losses.

Print compact epoch logs with train loss, validation loss, and elapsed time. Save the best validation checkpoint and reload it for final evaluation. Record seed, architecture, optimizer settings, best epoch, train/validation curves, device, parameter count, data hash, and measured duration.

Save `artifacts/model.pt` with a state dict and primitive metadata: tokenizer, special IDs, architecture, format version, and training metadata. Do not serialize the full model class. Save curves as JSON. Load safely with state-dict loading supported by the installed PyTorch version and map to the requested device. Check metadata compatibility.

Missing checkpoint should produce a concise instruction to run training. Explicit unavailable CUDA should fail clearly; automatic device selection should fall back to CPU. Exact reproducibility across hardware is not promised.

Check the local runtime before installing anything. If installation commands or versions are needed, verify against current official PyTorch documentation. Prefer CPU installation unless CUDA is already usable or explicitly desired; do not fetch a large GPU build speculatively.

## Step 6: Generate text autoregressively

Start with `[BOS, USER] + encoded_question + [ASSISTANT]`. Run the model, take final-position logits, select the next token, append it, and repeat until EOS or the generation cap.

- Default maximum new tokens: 160.
- Support temperature and top-k sampling. Validate arguments; top-k is bounded by eligible vocabulary size.
- Temperature zero means greedy decoding with argmax; do not divide by zero.
- Positive temperature divides logits before softmax.
- Forbid generation of PAD, BOS, USER, and ASSISTANT, and preferably UNK; allow EOS.
- Apply top-k by masking other logits to negative infinity, then sample with `torch.multinomial`.
- Offer an optional seed for reproducible demonstrations on a given environment.
- Crop the model input to the most recent 256 tokens when needed, with position IDs reset consistently. Document this simple sliding-window limitation.
- Use `eval()` and inference mode.
- Print only generated answer characters, not the prompt or role markers.
- Return a clear stop reason in explanation mode: EOS or length cap.

Do not add repetition penalties initially. If generation repeats, show it honestly and explain the limits instead of accumulating heuristics.

## Step 7: Inspect attention and learned weights

`python chat.py --explain` should show a compact educational diagnostic after generation:

- Actual parameter count, model dimensions, and device.
- The first next-token prediction's top five eligible tokens and probabilities, clearly labeled as a distribution over next tokens rather than answer confidence.
- Last-block attention from the final prompt position to prompt tokens, averaged over heads, before attention dropout in evaluation mode. List the strongest positions and weights with character/special-token labels and positions; repeated characters must remain distinguishable.
- Generation settings, generated-token count, and stop reason.

Explain that attention matrices are learned input-dependent relationships, whereas Q/K/V matrices are trainable parameters. Attention values do not prove causal importance or correctness. No chart library or image generation is required.

## Step 8: Evaluate honestly

`evaluate.py` loads the best saved checkpoint without retraining or modifying settings and writes `artifacts/evaluation.json`.

Report:

- Held-out token-average next-token cross-entropy and perplexity `exp(loss)`; handle numerical overflow explicitly.
- Number of examples, valid target tokens, unknown-character fraction, and split identity.
- Vocabulary size, unique parameter count, and checkpoint bytes.
- Measured generation tokens per second with device, warmup, generation count, prompt length, and timing scope. Synchronize CUDA around timers if used. Do not mix printing time into model timing without saying so.
- A fixed handful of held-out prompts with generated answers and reference answers, labeled as qualitative examples. Use greedy decoding for repeatable comparisons.
- Loss curves or their JSON path, noting obvious overfitting when training and validation diverge.

Do not use classifier accuracy for this model. Do not present exact-match answer accuracy or subjective examples as broad conversational competence. If adding answer-only loss, label it separately and use the correct answer-position token count.

Test examples are for final reporting, not iterative parameter selection. If repeatedly reviewing test outputs to change the model, disclose that the set has effectively become validation and reserve a fresh final holdout.

## Step 9: Meaningful tests and smoke checks

Use standard-library unittest with tiny synthetic fixtures. Verify:

1. Tokenizer round-trip on known text, stable special IDs, and unknown-character behavior.
2. Vocabulary is built from training only; split families and duplicate constraints are enforced.
3. Shifted input/target alignment and padding ignored in loss.
4. Forward shapes and finite logits/loss at sequence length 1 and normal lengths.
5. Causal attention weights above the diagonal are zero.
6. Perturbing future tokens does not change earlier logits in evaluation mode.
7. Padding keys receive zero attention for valid queries; a short example produces matching valid logits alone and padded beside a longer example, within numeric tolerance.
8. Attention weights sum to one over valid keys in evaluation mode.
9. Finite, nonzero gradients reach token embeddings, Q/K/V projections, and output computation on a nondegenerate fixture.
10. A very small synthetic next-token task can be overfit in a bounded deterministic CPU check; keep this separate or lightweight enough not to slow ordinary tests unnecessarily.
11. Checkpoint round-trip preserves evaluation logits and tokenizer mappings.
12. Greedy generation is deterministic, special-token restrictions work, EOS stops generation, and length caps terminate generation.
13. The actual configured model stays below the parameter budget.

Run all focused tests on CPU. Run a smoke training pass to catch pipeline errors, then one actual default or clearly documented bounded training run, held-out evaluation, and scripted chat checks including empty input, a known-style prompt, an unfamiliar prompt, explain mode, help, and quit. Test CUDA only if available and report when it was not tested. Do not make expensive repeated runs without evidence they are necessary.

If dependencies or compute block training, finish independent implementation work and clearly state which verification remains incomplete. A successful smoke pass is not a completed full training run.

## Step 10: Documentation and interview explanation

Write a concise but substantial README with:

- Educational purpose and limitations stated immediately.
- Setup using a virtual environment, Windows PowerShell commands, CPU-first installation, and optional official CUDA installation guidance.
- Exact commands for prepare, train, evaluate, chat, explain, and tests.
- Corpus format, fictional fact sheet, split logic, training-only vocabulary, and leakage precautions.
- Architecture diagram and tensor dimensions.
- A connected explanation of tokenization -> embeddings + positions -> Q/K/V -> causal attention -> residual/MLP blocks -> logits -> next-token cross-entropy -> gradients/optimizer -> saved weights -> autoregressive sampling.
- Difference between trainable parameters, activations, and attention weights.
- Why training can process all next-token positions in parallel but generation proceeds one token at a time.
- Why this character model is small, why it is not pretrained, and why generated quality is limited.
- Actual parameter count, training conditions, held-out loss/perplexity, checkpoint size, and measured speed; mark unavailable measurements honestly.
- Real example outputs from the completed model, including a failure or limitation.
- Temperature/top-k behavior, context truncation, and attention explanation caveats.
- No claims of production readiness, reasoning ability, robust factuality, or calibrated answer confidence.
- A truthful suggested resume bullet, without invented metrics or dates: "Implemented a tiny decoder-only language model in PyTorch from scratch, including causal multihead attention, next-token training, checkpointing, and autoregressive text generation; evaluated held-out loss and perplexity."
- Short interview talking points: why causal masking, why divide by square root of head dimension, why position embeddings, what loss optimizes, how gradients update weights, what overfitting looks like, and how temperature changes sampling.

No feedback database or continual-learning mechanism is required. Describe manual dataset revision and validation discipline as a possible future extension, without pretending that chat inputs automatically train the model.

## Implementation order and completion criteria

1. Inspect workspace and instructions.
2. Author fixed corpus and preparation checks.
3. Implement tokenizer, batching, transformer, and focused masking tests.
4. Implement training and checkpointing.
5. Implement generation and terminal interaction.
6. Implement explanation and evaluation.
7. Run tests, actual training, evaluation, and CLI smoke checks; fix observed errors.
8. Document measured results and limitations.
9. Review scope, dependencies, data leakage, and claims.

Done means the project can prepare local data, train its own small transformer, save/reload learned weights, generate new answer text token by token, inspect attention, and report held-out language-model metrics. Correct mechanics and transparent limitations matter more than fluent answers. No retrieval or canned-response shortcut satisfies this task.

Final response: link the main files, summarize what was implemented, report actual parameter count and measured evaluation results, identify checks performed, and disclose any unverified behavior. Do not claim completion if necessary training or verification is still blocked.

