"""历史编码器与辅助损失（issue #5）的单元测试与性能基准。"""

import gc
import time
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from ipd.agents.encoder import (
        HistoryEncoder,
        NEXT_MOVE_TARGET,
        moves_to_indices,
        observation_tensors,
    )
    from ipd.training.auxiliary import (
        NextMovePredictor,
        combined_auxiliary_loss,
        contrastive_loss,
        history_contrastive_loss,
        next_move_loss,
    )

TURNS = 25
HISTORY = 20


def history(batch=1, hidden=0.0, seed=0):
    """随机 c/b 历史；hidden 为逐步隐藏的概率。"""
    generator = torch.Generator().manual_seed(seed)
    own = torch.randint(1, 3, (batch, HISTORY, TURNS), generator=generator)
    opp = torch.randint(1, 3, (batch, HISTORY, TURNS), generator=generator)
    if hidden:
        mask = torch.rand((batch, HISTORY, TURNS), generator=generator) < hidden
        own = own.masked_fill(mask, 0)
        opp = opp.masked_fill(mask, 0)
    return own, opp


def current(batch=1, length=24, seed=1):
    generator = torch.Generator().manual_seed(seed)
    my = torch.randint(1, 3, (batch, length), generator=generator)
    opp = torch.randint(1, 3, (batch, length), generator=generator)
    return my, opp


@unittest.skipIf(torch is None, "torch not installed")
class EncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = HistoryEncoder().eval()

    def test_forward_shapes(self):
        for batch in (1, 32):
            with self.subTest(batch=batch):
                own, opp = history(batch)
                my, oppd = current(batch)
                with torch.no_grad():
                    out = self.model(own, opp, my, oppd, 0, 24)
                    pooled = self.model.encode_history(own, opp)
                self.assertEqual(out.shape, (batch, 32))
                self.assertEqual(pooled.shape, (batch, 32))
                self.assertTrue(torch.isfinite(out).all())

    def test_empty_history(self):
        own = torch.zeros(2, HISTORY, TURNS, dtype=torch.long)
        opp = torch.zeros(2, HISTORY, TURNS, dtype=torch.long)
        my = torch.zeros(2, 0, dtype=torch.long)
        oppd = torch.zeros(2, 0, dtype=torch.long)
        with torch.no_grad():
            pooled = self.model.encode_history(own, opp)
            out = self.model(own, opp, my, oppd, 0, 0)
            matches, valid = self.model.encode_matches(own, opp)
            pool_zero, weights = self.model.pool_matches(
                matches, valid, return_weights=True
            )
        self.assertTrue(torch.equal(pooled, torch.zeros_like(pooled)))
        self.assertTrue(torch.equal(pool_zero, torch.zeros_like(pool_zero)))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(weights).all())

    def test_full_and_partial_history(self):
        for hidden in (0.0, 0.1, 0.5, 0.9):
            with self.subTest(hidden=hidden):
                own, opp = history(2, hidden=hidden, seed=3)
                my, oppd = current(2)
                with torch.no_grad():
                    out = self.model(own, opp, my, oppd, 2, 20)
                self.assertTrue(torch.isfinite(out).all())

    def test_deterministic_in_eval(self):
        own, opp = history(2)
        my, oppd = current(2)
        with torch.no_grad():
            first = self.model(own, opp, my, oppd, 0, 24)
            second = self.model(own, opp, my, oppd, 0, 24)
        self.assertTrue(torch.equal(first, second))

    def test_batch_invariance(self):
        own, opp = history(8)
        my, oppd = current(8)
        with torch.no_grad():
            batched = self.model(own, opp, my, oppd, 0, 24)
            singles = torch.cat(
                [
                    self.model(
                        own[i:i + 1],
                        opp[i:i + 1],
                        my[i:i + 1],
                        oppd[i:i + 1],
                        0,
                        24,
                    )
                    for i in range(8)
                ],
                dim=0,
            )
        self.assertTrue(torch.allclose(batched, singles, atol=1e-5, rtol=1e-4))

    def test_one_hot_ngram_and_masking(self):
        own = torch.zeros(1, HISTORY, TURNS, dtype=torch.long)
        opp = torch.zeros(1, HISTORY, TURNS, dtype=torch.long)
        # 第 19 场第 0 手：c/c，阶段 0 -> 索引 1*9 + 1*3 + 0 = 12
        own[0, 19, 0] = 1
        opp[0, 19, 0] = 1
        # 第 18 场第 10 手：c/b，阶段 1 -> 索引 1*9 + 2*3 + 1 = 16
        own[0, 18, 10] = 1
        opp[0, 18, 10] = 2
        with torch.no_grad():
            matches, valid = self.model.encode_matches(own, opp)
        self.assertEqual(valid[0].sum().item(), 2)
        self.assertFalse(valid[0, 17])
        self.assertTrue(torch.allclose(matches[0, 19], self.model.embed.weight[12]))
        self.assertTrue(torch.allclose(matches[0, 18], self.model.embed.weight[16]))
        self.assertTrue(torch.equal(matches[0, 0], torch.zeros_like(matches[0, 0])))

    def test_attention_masks_hidden_matches(self):
        matches = torch.ones(1, HISTORY, 32)
        valid = torch.zeros(1, HISTORY, dtype=torch.bool)
        valid[0, -5:] = True
        with torch.no_grad():
            pooled, weights = self.model.pool_matches(
                matches, valid, return_weights=True
            )
        self.assertEqual(tuple(weights.shape), (1, 1, HISTORY))
        self.assertLessEqual(weights[0, 0, :-5].abs().max().item(), 1e-5)
        self.assertAlmostEqual(weights[0, 0, -5:].sum().item(), 1.0, places=5)
        self.assertTrue(torch.isfinite(pooled).all())

    def test_time_decay_prefers_recent_matches(self):
        matches = torch.ones(1, HISTORY, 32)
        valid = torch.ones(1, HISTORY, dtype=torch.bool)
        with torch.no_grad():
            _, weights = self.model.pool_matches(matches, valid, return_weights=True)
        decay = weights[0, 0]
        for i in range(HISTORY - 1):
            self.assertLessEqual(decay[i].item(), decay[i + 1].item() + 1e-7)
        self.assertGreater(decay[-1].item(), decay[0].item())

    def test_length_zero(self):
        own, opp = history(2)
        my = torch.zeros(2, 0, dtype=torch.long)
        oppd = torch.zeros(2, 0, dtype=torch.long)
        with torch.no_grad():
            out = self.model(own, opp, my, oppd, 3, 0)
        self.assertEqual(out.shape, (2, 32))
        self.assertTrue(torch.isfinite(out).all())

    def test_round_conditioning_changes_output(self):
        own, opp = history(1)
        my, oppd = current(1)
        with torch.no_grad():
            round0 = self.model(own, opp, my, oppd, 0, 24)
            round7 = self.model(own, opp, my, oppd, 7, 24)
        self.assertFalse(torch.allclose(round0, round7))

    def test_gradients_flow(self):
        self.model.train()
        own, opp = history(2)
        my, oppd = current(2)
        out = self.model(own, opp, my, oppd, 0, 24)
        out.pow(2).mean().backward()
        for name, parameter in self.model.named_parameters():
            if name == "round_embed.weight":
                # rs=0 时只有第 0 行被使用；其余参数都必须有梯度。
                continue
            self.assertIsNotNone(parameter.grad, name)
        self.assertGreater(self.model.embed.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(self.model.gru.weight_ih_l0.grad.abs().sum().item(), 0.0)

    def test_validation_errors(self):
        own, opp = history(2)
        my, oppd = current(2)
        with self.assertRaises(ValueError):
            self.model(
                torch.zeros(2, 2, 2, dtype=torch.long),
                torch.zeros(2, 2, 2, dtype=torch.long),
                my,
                oppd,
                0,
                24,
            )
        with self.assertRaises(ValueError):
            self.model(own, opp[:, :, :10], my, oppd, 0, 24)
        with self.assertRaises(TypeError):
            self.model(own.float(), opp, my, oppd, 0, 24)
        with self.assertRaises(ValueError):
            self.model(own, opp, my, oppd, 0, 25)
        with self.assertRaises(ValueError):
            self.model(own, opp, my[:, :10], oppd, 0, 24)
        with self.assertRaises(ValueError):
            self.model(own, opp, my, oppd, 8, 24)

    def test_observation_tensors(self):
        rows = tuple(tuple("cb"[i % 2] for i in range(TURNS)) for _ in range(HISTORY))
        own, opp, my, oppd, rs = observation_tensors(
            rows, rows, ("c", "b"), ("b", "c"), 3
        )
        self.assertEqual(tuple(own.shape), (1, HISTORY, TURNS))
        self.assertEqual(tuple(my.shape), (1, 2))
        self.assertEqual(rs.tolist(), [3])
        self.assertEqual(my.tolist(), [[1, 2]])
        self.assertEqual(oppd.tolist(), [[2, 1]])
        self.assertTrue(torch.equal(own, opp))

    def test_num_parameters(self):
        self.assertGreater(self.model.num_parameters(), 0)

    def test_performance_one_millisecond(self):
        """CPU、batch=1、完整历史：100 次前向的最坏耗时 ≤ 1 ms。"""
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            model = HistoryEncoder().eval()
            own, opp = history(1)
            my, oppd = current(1)
            with torch.inference_mode():
                for _ in range(50):
                    model(own, opp, my, oppd, 0, 24)
                times = []
                gc.disable()
                try:
                    for _ in range(100):
                        start = time.perf_counter()
                        model(own, opp, my, oppd, 0, 24)
                        times.append(time.perf_counter() - start)
                finally:
                    gc.enable()
            times.sort()
            worst, median = times[-1], times[len(times) // 2]
            print(
                f"\nencoder latency (batch=1, 20x25 history): "
                f"worst {worst * 1e3:.3f} ms, median {median * 1e3:.3f} ms"
            )
            self.assertLess(worst, 1e-3)
        finally:
            torch.set_num_threads(old_threads)


@unittest.skipIf(torch is None, "torch not installed")
class AuxiliaryLossTest(unittest.TestCase):
    def test_next_move_predictor_shape_and_loss(self):
        torch.manual_seed(0)
        head = NextMovePredictor(16)
        features = torch.randn(8, 16)
        logits = head(features)
        self.assertEqual(tuple(logits.shape), (8, 2))
        target = torch.randint(0, 2, (8,))
        loss = next_move_loss(logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(head.net[0].weight.grad)

    def test_next_move_head_learns(self):
        torch.manual_seed(0)
        head = NextMovePredictor(8, hidden_dim=16)
        features = torch.randn(64, 8)
        target = (features[:, 0] > 0).long()
        optimizer = torch.optim.Adam(head.parameters(), lr=0.05)
        with torch.no_grad():
            initial = next_move_loss(head(features), target).item()
        for _ in range(80):
            optimizer.zero_grad()
            loss = next_move_loss(head(features), target)
            loss.backward()
            optimizer.step()
        final = next_move_loss(head(features), target).item()
        self.assertLess(final, initial)

    def test_contrastive_prefers_similar_views(self):
        torch.manual_seed(0)
        anchor = torch.randn(8, 16, requires_grad=True)
        similar = anchor + 0.01 * torch.randn(8, 16)
        different = torch.randn(8, 16)
        same_loss = contrastive_loss(anchor, similar)
        different_loss = contrastive_loss(anchor, different)
        self.assertTrue(torch.isfinite(same_loss))
        self.assertLess(same_loss.item(), different_loss.item())
        same_loss.backward()

    def test_contrastive_batch_one_and_bad_shapes(self):
        one = torch.randn(1, 4)
        loss = contrastive_loss(one, one)
        self.assertEqual(loss.item(), 0.0)
        with self.assertRaises(ValueError):
            contrastive_loss(torch.randn(2, 4), torch.randn(2, 5))

    def test_history_contrastive_loss(self):
        torch.manual_seed(0)
        model = HistoryEncoder().eval()
        own_a, opp_a = history(4, seed=0)
        own_b, opp_b = history(4, seed=1)
        loss = history_contrastive_loss(model, (own_a, opp_a), (own_b, opp_b))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.embed.weight.grad)

    def test_combined_auxiliary_loss(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 2, requires_grad=True)
        target = torch.tensor([0, 1, 1, 0])
        anchor = torch.randn(4, 8, requires_grad=True)
        positive = torch.randn(4, 8, requires_grad=True)
        loss = combined_auxiliary_loss(
            logits, target, anchor, positive, next_weight=1.0, contrastive_weight=0.5
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(anchor.grad)


if __name__ == "__main__":
    unittest.main()
