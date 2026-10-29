"""Mechanics tests use synthetic fixtures, never final test-set outputs."""
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

import chat
from model import Config, TinyLM, next_token_loss
from prepare import SPECIAL, Tokenizer, answer_targets, batchify, prepare, validate_records
from utils import device_for, load_checkpoint, measure_loss, save_checkpoint, seed_all
from train import training_examples


def fixture_records():
    return [dict(split=s, family=s, question=q, answer=a) for s, q, a in
            [("train", "abc?", "cab."), ("validation", "z?", "a."), ("test", "x?", "b.")]]


class LanguageModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        seed_all(123)
        self.tokenizer = Tokenizer.from_records(fixture_records())
        self.model = TinyLM(Config(len(self.tokenizer.labels), context=32, width=24,
                                   layers=2, heads=4, dropout=0)).eval()

    def test_tokenizer_roundtrip_special_ids_and_unknown(self):
        text = "abc?cab."
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)
        self.assertEqual(SPECIAL, {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<user>": 3, "<assistant>": 4, "<eos>": 5})
        self.assertEqual(self.tokenizer.encode("z"), [1])
        self.assertEqual(self.tokenizer.decode([2, 3, 1, 4, 5, 0]), "\ufffd")
        self.assertEqual(Tokenizer.from_metadata(self.tokenizer.metadata()).to_id, self.tokenizer.to_id)
        bad = self.tokenizer.metadata() | {"special_ids": {}}
        with self.assertRaisesRegex(ValueError, "special"):
            Tokenizer.from_metadata(bad)

    def test_question_normalization_matches_training_and_chat(self):
        question = "  ABC?  "
        recorded = self.tokenizer.record(question, "cab.")
        prompt, _ = chat.prompt_tokens(self.tokenizer, question, 32)
        self.assertEqual(recorded[:recorded.index(4) + 1], prompt)
        self.assertEqual(prompt, chat.prompt_tokens(self.tokenizer, "abc?", 32)[0])
        self.assertEqual(prompt, chat.prompt_tokens(self.tokenizer, "AbC", 32)[0])
        self.assertEqual(prompt, chat.prompt_tokens(self.tokenizer, "abc!!", 32)[0])
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode("ABC")), "\ufffd\ufffd\ufffd")

    def test_splits_vocabulary_and_duplicates(self):
        rows = fixture_records()
        validate_records(rows)
        self.assertNotIn("z", self.tokenizer.to_id)
        self.assertNotIn("x", self.tokenizer.to_id)
        bad = copy.deepcopy(rows)
        bad[1]["family"] = "train"
        with self.assertRaisesRegex(ValueError, "spans splits"):
            validate_records(bad)
        bad = copy.deepcopy(rows)
        bad[1].update(question="ABC ?", answer="CAB !")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_records(bad)

    def test_prepare_rejects_overlong_and_persists(self):
        with tempfile.TemporaryDirectory() as folder:
            data = prepare(fixture_records(), Path(folder))
            self.assertEqual(len(data["splits"]["train"]), 1)
            bad = fixture_records()
            bad[0]["question"] = "a" * 256
            with self.assertRaisesRegex(ValueError, "shorten"):
                prepare(bad, Path(folder))

    def test_shift_alignment_and_ignored_padding(self):
        rows = [[2, 3, 7, 4, 8, 5], [2, 3, 4, 5]]
        x, y = batchify(rows)
        self.assertEqual(x[0].tolist(), rows[0][:-1])
        self.assertEqual(y[0].tolist(), rows[0][1:])
        self.assertEqual(y[1].tolist(), [3, 4, 5, 0, 0])
        logits = torch.randn(2, 5, len(self.tokenizer.labels))
        original = next_token_loss(logits, y)
        logits[1, 3:] = 1000 * torch.randn_like(logits[1, 3:])
        torch.testing.assert_close(original, next_token_loss(logits, y))
        manual = torch.nn.functional.cross_entropy(logits[y != 0], y[y != 0])
        torch.testing.assert_close(original, manual)

    def test_forward_shapes_finite_at_one_and_normal_lengths(self):
        for length in (1, 12, 32):
            x = torch.full((2, length), 6, dtype=torch.long)
            logits, attention = self.model(x)
            self.assertEqual(logits.shape, (2, length, len(self.tokenizer.labels)))
            self.assertIsNone(attention)
            self.assertTrue(torch.isfinite(logits).all())
            self.assertTrue(torch.isfinite(next_token_loss(logits, x)))
        with self.assertRaises(ValueError):
            self.model(torch.ones(1, 33, dtype=torch.long))

    def test_answer_loss_alignment_and_token_weighting(self):
        rows = [self.tokenizer.record("a?", "cab."), self.tokenizer.record("abc?", "b.")]
        x, y = batchify(rows)
        masked = answer_targets(x, y)
        for i, row in enumerate(rows):
            start = row.index(4)
            self.assertTrue(masked[i, :start].eq(0).all())
            self.assertEqual(masked[i, start:len(row) - 1].tolist(), row[start + 1:])
        self.assertEqual(masked.ne(0).sum().item(), 4 + 1 + 2 + 1)
        loss, count = measure_loss(self.model, rows, 1, torch.device("cpu"), answer_only=True)
        expected = next_token_loss(self.model(x)[0], masked).item()
        self.assertAlmostEqual(loss, expected, places=6)
        self.assertEqual(count, 8)

    def test_training_augmentation_preserves_answer_and_context_limit(self):
        rows = fixture_records()[:1] * 20
        examples = training_examples(rows, self.tokenizer, augment=True)
        original = self.tokenizer.record(rows[0]["question"], rows[0]["answer"])
        self.assertTrue(any(len(row) > len(original) for row in examples))
        for row in examples:
            self.assertLessEqual(len(row), 257)
            self.assertEqual(row[row.index(4):], original[original.index(4):])
            self.assertEqual(row[:2], [2, 3])
        self.assertEqual(training_examples(rows, self.tokenizer, augment=False), [original] * 20)
        default = TinyLM(Config(len(self.tokenizer.labels), width=128, position_jitter=True))
        self.assertLess(default.parameter_count(), 500_000)

    def test_causality_and_attention_normalization(self):
        x = torch.tensor([[2, 3, 6, 7, 8, 4]])
        logits, attention = self.model(x, inspect=True)
        for weights in attention:
            self.assertEqual(weights.triu(1).count_nonzero().item(), 0)
            torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
        altered = x.clone()
        altered[0, 3:] = torch.tensor([9, 6, 5])
        other, _ = self.model(altered)
        torch.testing.assert_close(logits[:, :3], other[:, :3], atol=1e-6, rtol=1e-5)

    def test_position_jitter_is_training_only_and_stays_in_bounds(self):
        model = TinyLM(Config(len(self.tokenizer.labels), context=16, width=16, heads=2,
                              layers=1, dropout=0, position_jitter=True))
        x = torch.tensor([[2, 3, 6, 4], [2, 3, 7, 4]])
        model.train()
        a, _ = model(x)
        b, _ = model(x)
        self.assertFalse(torch.equal(a, b))
        self.assertTrue(torch.isfinite(a).all())
        model(torch.full((2, 16), 6, dtype=torch.long))  # Zero offset at full context.
        model.eval()
        torch.testing.assert_close(model(x)[0], model(x)[0], atol=0, rtol=0)

    def test_padding_keys_and_batch_invariance(self):
        short = torch.tensor([[2, 3, 6, 4]])
        padded = torch.tensor([[2, 3, 6, 4, 0, 0], [2, 3, 8, 7, 6, 4]])
        alone, _ = self.model(short)
        together, attention = self.model(padded, inspect=True)
        torch.testing.assert_close(alone[0], together[0, :4], atol=1e-6, rtol=1e-5)
        for weights in attention:
            self.assertEqual(weights[0, :, :4, 4:].count_nonzero().item(), 0)
        # Internal/leading padding must also never become a valid query's key.
        strange = torch.tensor([[0, 2, 0, 3]])
        logits, attention = self.model(strange, inspect=True)
        self.assertTrue(torch.isfinite(logits).all())
        for weights in attention:
            self.assertEqual(weights[0, :, 3, [0, 2]].count_nonzero().item(), 0)

    def test_gradients_reach_every_requested_component(self):
        x, y = batchify([self.tokenizer.record("abc?", "cab."), self.tokenizer.record("c?", "ab.")])
        logits, _ = self.model(x)
        logits.retain_grad()
        next_token_loss(logits, y).backward()
        names = ["token_embedding.weight", "blocks.0.attention.query.weight",
                 "blocks.0.attention.key.weight", "blocks.0.attention.value.weight"]
        for name, parameter in self.model.named_parameters():
            if name in names:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        self.assertIs(self.model.output.weight, self.model.token_embedding.weight)

    def test_small_task_overfits_on_cpu(self):
        model = TinyLM(Config(8, context=8, width=16, heads=2, layers=1, dropout=0))
        x, y = batchify([[2, 3, 6, 4, 7, 5], [2, 3, 7, 4, 6, 5]])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.025, weight_decay=0)
        initial = next_token_loss(model(x)[0], y).item()
        for _ in range(120):
            optimizer.zero_grad(set_to_none=True)
            loss = next_token_loss(model(x)[0], y)
            loss.backward()
            optimizer.step()
        final = next_token_loss(model(x)[0], y).item()
        # First question character is ambiguous; irreducible average ~log(2)/5.
        self.assertLess(final, 0.22)
        self.assertLess(final, initial / 5)

    def test_checkpoint_roundtrip_and_compatibility(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.pt"
            save_checkpoint(path, self.model, self.tokenizer, {"seed": 123})
            restored, tokenizer, _ = load_checkpoint(torch.device("cpu"), path)
            x = torch.tensor([[2, 3, 6, 4]])
            torch.testing.assert_close(self.model(x)[0], restored(x)[0], atol=0, rtol=0)
            self.assertEqual(tokenizer.to_id, self.tokenizer.to_id)
            payload = torch.load(path, weights_only=True)
            payload["format_version"] = 999
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "format"):
                load_checkpoint(torch.device("cpu"), path)

    def test_generation_determinism_restrictions_stops_and_caps(self):
        first = chat.generate(self.model, self.tokenizer, "abc?", temperature=0, max_new_tokens=8)
        second = chat.generate(self.model, self.tokenizer, "abc?", temperature=0, max_new_tokens=8)
        self.assertEqual(first, second)
        self.assertTrue(all(i >= 5 for i in first["token_ids"]))
        class FixedLM(torch.nn.Module):
            def __init__(self, token):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.config = Config(12, context=8)
                self.token = token
            def forward(self, x, inspect=False):
                assert x.shape[1] <= 8
                logits = torch.zeros(1, x.shape[1], 12)
                logits[..., :5] = 100  # Forbidden tokens have the largest raw logits.
                logits[..., self.token] = 50
                return logits, None
        eos = chat.generate(FixedLM(5), self.tokenizer, "abc?", temperature=0)
        self.assertEqual(eos["token_ids"], [5])
        self.assertEqual(eos["answer"], "")
        self.assertEqual(eos["stop_reason"], "EOS")
        cap = chat.generate(FixedLM(6), self.tokenizer, "abc?" * 30, temperature=0, max_new_tokens=20)
        self.assertEqual(cap["generated_tokens"], 20)
        self.assertEqual(cap["stop_reason"], "length cap")
        self.assertTrue(cap["truncated"])
        ids, truncated = chat.prompt_tokens(self.tokenizer, "a" * 40, 8)
        self.assertEqual((ids[:2], ids[-1], len(ids), truncated), ([2, 3], 4, 8, True))

    def test_sampling_settings_and_explanation(self):
        logits = torch.arange(12, dtype=torch.float32)
        probs = chat.sampling_distribution(logits, 0.7, 2)
        self.assertEqual(probs.count_nonzero().item(), 2)
        self.assertEqual(probs[:5].sum().item(), 0)
        torch.testing.assert_close(probs.sum(), torch.tensor(1.0))
        self.assertEqual(chat.sampling_distribution(logits, 1, 999).count_nonzero().item(), 7)
        for t, k in [(-1, 0), (float("nan"), 2), (1, -1)]:
            with self.assertRaises(ValueError):
                chat.sampling_distribution(logits, t, k)
        out = chat.generate(self.model, self.tokenizer, "a?", temperature=0, max_new_tokens=2, explain=True)
        self.assertEqual(len(out["diagnostic"]["next_tokens"]), 5)
        self.assertTrue(out["diagnostic"]["attention"])

    def test_token_weighted_evaluation(self):
        rows = [self.tokenizer.record("a?", "b."), self.tokenizer.record("abc?", "cab.")]
        loss, count = measure_loss(self.model, rows, 1, torch.device("cpu"))
        x, y = batchify(rows)
        expected = next_token_loss(self.model(x)[0], y).item()
        self.assertAlmostEqual(loss, expected, places=6)
        self.assertEqual(count, sum(len(r) - 1 for r in rows))

    def test_budget_and_device(self):
        default = TinyLM(Config(len(self.tokenizer.labels)))
        self.assertLess(default.parameter_count(), 500_000)
        self.assertGreater(default.parameter_count(), 150_000)
        self.assertEqual(device_for("cpu").type, "cpu")
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(device_for().type, "cpu")
            with self.assertRaisesRegex(ValueError, "unavailable"):
                device_for("cuda")

    def test_terminal_empty_help_quit_eof_and_interrupt(self):
        for ending in ("/quit", EOFError(), KeyboardInterrupt()):
            with patch.object(sys, "argv", ["chat.py"]), patch("chat.load_checkpoint", return_value=(self.model, self.tokenizer, {})), patch("builtins.input", side_effect=["", "/help", ending]), patch("chat.generate") as generate, patch("sys.stdout", new_callable=io.StringIO) as output:
                chat.main()
                generate.assert_not_called()
                self.assertIn("No memory", output.getvalue())


if __name__ == "__main__":
    unittest.main()
