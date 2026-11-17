"""策略网络（issue #6，sub-issue 8.1）的单元测试。"""

import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from ipd.pool import make_agent
    from ipd.sim import TournamentConfig, run_tournament
    from ipd.training.policy import (
        COOPERATE_INDEX,
        DEFECT_INDEX,
        IPDPolicy,
        action_to_move,
        move_to_action,
        policy_agent,
    )


def inputs(batch=2, length=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    own = torch.randint(0, 3, (batch, 20, 25), generator=generator)
    opp = torch.randint(0, 3, (batch, 20, 25), generator=generator)
    my = torch.randint(1, 3, (batch, length), generator=generator)
    their = torch.randint(1, 3, (batch, length), generator=generator)
    rs = torch.randint(0, 8, (batch,), generator=generator)
    return own, opp, my, their, rs, length


@unittest.skipIf(torch is None, "torch not installed")
class PolicyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.policy = IPDPolicy()

    def test_action_mapping(self):
        self.assertEqual(action_to_move(0), "c")
        self.assertEqual(action_to_move(1), "b")
        self.assertEqual(move_to_action("c"), COOPERATE_INDEX)
        self.assertEqual(move_to_action("b"), DEFECT_INDEX)
        with self.assertRaises((IndexError, ValueError)):
            action_to_move(2)

    def test_forward_shapes(self):
        for batch in (1, 4):
            with self.subTest(batch=batch):
                logits, values = self.policy.forward_with_value(*inputs(batch))
                self.assertEqual(logits.shape, (batch, 2))
                self.assertEqual(values.shape, (batch,))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.isfinite(values).all())

    def test_act_deterministic(self):
        args = inputs(8)
        with torch.no_grad():
            action, probs = self.policy.act(*args, deterministic=True)
            expected = probs.argmax(dim=-1)
        self.assertTrue(torch.equal(action, expected))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(8)))

    def test_act_sampling_is_legal_and_reproducible(self):
        args = inputs(16)
        with torch.no_grad():
            first, _ = self.policy.act(
                *args, generator=torch.Generator().manual_seed(123)
            )
            second, _ = self.policy.act(
                *args, generator=torch.Generator().manual_seed(123)
            )
            other, _ = self.policy.act(
                *args, generator=torch.Generator().manual_seed(999)
            )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(all(int(value) in (0, 1) for value in first))
        self.assertFalse(torch.equal(first, other))

    def test_act_with_value(self):
        args = inputs(3, length=0)
        with torch.no_grad():
            action, probs, values = self.policy.act_with_value(*args)
        self.assertEqual(action.shape, (3,))
        self.assertEqual(probs.shape, (3, 2))
        self.assertEqual(values.shape, (3,))

    def test_gradients_flow(self):
        logits = self.policy(*inputs(4))
        loss = logits[:, 0].sum()
        loss.backward()
        grads = [
            parameter.grad for parameter in self.policy.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))
        self.assertIsNotNone(self.policy.encoder.embed.weight.grad)

    def test_config_roundtrip(self):
        config = self.policy.config()
        recreated = IPDPolicy.from_config(config)
        recreated.load_state_dict(self.policy.state_dict())
        recreated.eval()
        self.policy.eval()
        args = inputs(2)
        with torch.no_grad():
            self.assertTrue(
                torch.allclose(self.policy(*args), recreated(*args), atol=1e-6)
            )
        self.assertEqual(
            self.policy.num_parameters(), recreated.num_parameters()
        )

    def test_rejects_bad_action_dim(self):
        with self.assertRaises(ValueError):
            IPDPolicy(action_dim=3)

    def test_policy_agent_is_deterministic_and_plays_cooperate(self):
        # 把动作头固定为"永远合作"：权重清零、偏置 [1, -1]。
        with torch.no_grad():
            self.policy.action_head.weight.zero_()
            self.policy.action_head.bias.copy_(torch.tensor([1.0, -1.0]))
        agent = policy_agent(self.policy, name="always_c")
        cfg = TournamentConfig(rounds=1, hide_prob=0.0, shuffle_schedule=False)
        result = run_tournament([agent, make_agent("all_d")], seed=0, cfg=cfg)
        # all_c vs all_d 的参考分是 -170（tests/test_sim.py）。
        self.assertEqual(result.scores[0], -170)
        self.assertEqual(result.faults, [0, 0])


if __name__ == "__main__":
    unittest.main()
