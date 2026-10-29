# Saved progress — 2026-10-29

Stopped at the user's request for manual judgment and a later iteration. No training or tuning is scheduled to continue.

## Run the saved model

From this project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe chat.py
```

Exit any already-running chat with `/quit` and restart it to load the new weights. `--explain` shows diagnostics. Greedy decoding is the default; `--temperature 0.5 --top-k 20 --seed 42` enables sampling. The optional NumPy warning remains harmless; no dependency was added to suppress it.

## Current result

- `artifacts/model.pt`: version 3, 438,272 unique parameters, width 128, two blocks, four heads, context 256. Actual autoregressive character generation remains intact; no retrieval, keyword routing, canned responses or external pretrained model.
- Both `What are the library hours?` and `what are the library Hours` generate: `The library opens at 8 am and closes at 8 pm on weekdays.` This answer omits weekend hours.
- Familiar short Wi-Fi, parking and tuition questions produced sensible answers. Greetings are readable. Occasional misspellings remain, such as `Chose Forgot password...`.
- Longer/unfamiliar questions often receive unrelated answers. One fresh parking question received password-reset instructions. **Readability improved for familiar prompts; generalization is still poor. Do not treat this as a reliable chatbot.**

## What changed

1. Shortened answers and expanded the fixed corpus to 390 examples: 300 train, 60 validation, 30 fresh test. The original test became validation after its failures informed development. No family crosses splits.
2. Added a correctly aligned answer-only loss and token count. Default training now optimizes answer characters and EOS; full-sequence metrics are still separately reported. `--objective all` remains available.
3. Increased training width from 96 to 128 while keeping the model under 500,000 parameters.
4. Added neutral question-prefix/suffix augmentation to vary input lengths. Experimental random position offsets were tried; they are off by default and were disabled in refinement.
5. Normalized question case, whitespace and terminal punctuation identically during training and inference. The tokenizer metadata and checkpoint version enforce compatibility. Token IDs were not remapped.
6. Added time budgets, a checkpoint-selection learning period, explicit continuation from a compatible local checkpoint, and preserved candidate checkpoints for comparison.
7. Defaulted chat to greedy decoding. No repetition penalty or fallback hides bad output.

## Training and selection caveat

Five CPU development runs took about 488 measured training seconds total. The saved model uses the final weights from the 150-epoch augmented width-128 run (211.21 seconds), followed by 30 refinement epochs (43.00 seconds) at learning rate 0.0003, with position offsets disabled. The initial weights were trained from scratch locally.

The lowest numerical validation-loss checkpoints were incoherent. Later candidates were compared on development/validation prompts. Refinement used a fixed 30-epoch budget, selecting at epoch 30. Thus the saved checkpoint is **not the global minimum-validation-loss checkpoint**. `best_epoch` is relative to the configured selection window. This tradeoff and the entire source/refinement history are recorded in the README and checkpoint metadata.

An RTX 4070 Laptop GPU with 8 GB VRAM was detected, but the installed PyTorch is CPU-only. No CUDA package was downloaded and no GPU training was used. These runs were short enough on CPU.

## Verification and measurements

- 19 CPU unit tests passed, including masking, causal isolation, gradients, checkpoint round-trip, synthetic overfit, generation limits, normalization, answer-only loss, and augmentation.
- Updated smoke training and scripted terminal checks passed, including the user's two question forms, explain mode, sampling, unknown input, empty/help/quit/EOF, truncation, and launch from another working directory. Ctrl+C is covered by the unit test's injected `KeyboardInterrupt`.
- Fresh final holdout evaluated once: all-token loss **5.2057**, perplexity **182.3084** on 3,853 targets; answer-only loss **4.4736**, perplexity **87.6761** on 1,764 answer/EOS targets.
- Checkpoint: 1,909,785 bytes. CPU generation: about 1,080.7 tokens/sec over ten fixed prompts after one warmup. See the report for timing scope.
- These metrics do not show good generalization. They are not directly comparable to the original baseline because the holdout and objective changed.

## Files to inspect next

- `README.md`: full setup, architecture, training decisions, actual results and limitations.
- `artifacts/evaluation.json`: frozen model's fresh-holdout metrics and all ten qualitative examples.
- `artifacts/candidate-generation-review.json`: familiar and validation prompt outputs for selected/final candidates; no fresh-test prompts were used in this comparison.
- `artifacts/cli-checks.json`: actual terminal transcripts.
- `artifacts/curves.json`: refinement metadata and nested source-run history.
- `artifacts/baseline-v1/`: original checkpoint, data and report.
- `artifacts/revision-all/`, `revision-jitter-all/`, `revision-answer/`, `revision-augmented/`: earlier development candidates.
- `artifacts/revision-augmented/last-model.pt`: source weights for the saved refinement. The README gives its exact continuation command.

All files are saved locally; generated artifacts and `.venv` are ignored by Git. There is no Git repository initialized here and no commit was made.

## Next iteration considerations

Start by judging actual answers, especially unfamiliar short paraphrases and questions that combine facts. Do not equate fluent answers with correct ones. The model still memorizes narrow answer patterns and confuses topics.

The fresh holdout has now been reported. If its failures inform further data/model changes, promote it to validation and author another final holdout. Preserve existing checkpoints before retraining. Additional training alone has not reliably improved generalization; consider whether to improve corpus diversity or relax the tiny, from-scratch model constraint before spending more compute. No further experiments are running.
