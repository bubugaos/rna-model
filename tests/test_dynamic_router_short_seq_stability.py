import unittest
import torch

from src.model.dynamic_router import DynamicRouter


class TestDynamicRouterShortSeqStability(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.dim = 16
        self.batch_size = 3
        self.router = DynamicRouter(dim=self.dim, beta=0.5)

    def _build_inputs(self, seq_len: int):
        h_local = torch.randn(self.batch_size, seq_len, self.dim)
        bppm = torch.rand(self.batch_size, seq_len, seq_len)
        bppm = (bppm + bppm.transpose(1, 2)) / 2
        # Compute cross_pair_sum the same way as _extract_bppm_vectors
        S = bppm.cumsum(dim=1).cumsum(dim=2)
        cross_pair_sum = (S[:, :, -1] - torch.diagonal(S, dim1=1, dim2=2))[:, :-1]
        return h_local, cross_pair_sum

    def test_len1_train_eval_stable(self):
        h_local, cross_pair_sum = self._build_inputs(seq_len=1)
        mask = torch.ones(self.batch_size, 1, dtype=torch.float32)

        self.router.train()
        out_train = self.router(h_local, cross_pair_sum=cross_pair_sum, mask=mask)
        self.router.eval()
        out_eval = self.router(h_local, cross_pair_sum=cross_pair_sum, mask=mask)

        for out in (out_train, out_eval):
            self.assertEqual(out["boundary_probs"].shape, (self.batch_size, 0))
            self.assertEqual(out["boundary_mask"].shape, (self.batch_size, 0))
            self.assertEqual(out["segment_ids"].shape, (self.batch_size, 1))
            self.assertTrue(torch.isfinite(out["expected_segments"]).all().item())

    def test_len2_with_padding_stable(self):
        h_local, cross_pair_sum = self._build_inputs(seq_len=2)
        mask = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        )

        self.router.train()
        out = self.router(h_local, cross_pair_sum=cross_pair_sum, mask=mask)

        self.assertEqual(out["boundary_probs"].shape, (self.batch_size, 1))
        self.assertTrue(torch.isfinite(out["boundary_probs"]).all().item())
        self.assertTrue(((out["boundary_probs"] >= 0.0) & (out["boundary_probs"] <= 1.0)).all().item())
        self.assertEqual(int(out["segment_ids"][1, 1].item()), -1)
    def test_beta_zero_disables_bppm_effect(self):
        torch.manual_seed(11)
        router = DynamicRouter(dim=self.dim, beta=0.0)
        h_local, _ = self._build_inputs(seq_len=8)
        mask = torch.ones(self.batch_size, 8, dtype=torch.float32)
        bppm_zero = torch.zeros(self.batch_size, 8, 8)
        bppm_strong = torch.zeros(self.batch_size, 8, 8)
        bppm_strong[:, 1, 6] = 0.95
        bppm_strong[:, 6, 1] = 0.95
        # Convert to cross_pair_sum
        def _to_cps(b):
            S = b.cumsum(dim=1).cumsum(dim=2)
            return (S[:, :, -1] - torch.diagonal(S, dim1=1, dim2=2))[:, :-1]
        cps_zero = _to_cps(bppm_zero)
        cps_strong = _to_cps(bppm_strong)

        router.eval()
        with torch.no_grad():
            out_zero = router(h_local, cross_pair_sum=cps_zero, mask=mask)
            out_strong = router(h_local, cross_pair_sum=cps_strong, mask=mask)

        self.assertTrue(torch.allclose(out_zero["boundary_logits"], out_strong["boundary_logits"], atol=1e-6))
        self.assertTrue(torch.allclose(out_zero["boundary_probs"], out_strong["boundary_probs"], atol=1e-6))

    def test_positive_beta_keeps_bppm_effect(self):
        torch.manual_seed(12)
        router = DynamicRouter(dim=self.dim, beta=1.0)
        h_local, _ = self._build_inputs(seq_len=8)
        mask = torch.ones(self.batch_size, 8, dtype=torch.float32)
        bppm_zero = torch.zeros(self.batch_size, 8, 8)
        bppm_strong = torch.zeros(self.batch_size, 8, 8)
        bppm_strong[:, 1, 6] = 0.95
        bppm_strong[:, 6, 1] = 0.95
        def _to_cps(b):
            S = b.cumsum(dim=1).cumsum(dim=2)
            return (S[:, :, -1] - torch.diagonal(S, dim1=1, dim2=2))[:, :-1]
        cps_zero = _to_cps(bppm_zero)
        cps_strong = _to_cps(bppm_strong)

        router.eval()
        with torch.no_grad():
            out_zero = router(h_local, cross_pair_sum=cps_zero, mask=mask)
            out_strong = router(h_local, cross_pair_sum=cps_strong, mask=mask)

        self.assertFalse(torch.allclose(out_zero["boundary_logits"], out_strong["boundary_logits"], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
