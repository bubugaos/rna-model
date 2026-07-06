import unittest

import torch

from src.model.chunking import Dechunker, Downsampler
from src.model.latent_transformer import LatentTransformer
from src.model.model import RNADynamicModel


class TestDynamicChunkingRefactor(unittest.TestCase):
    def test_dechunker_ema_smoothing(self):
        dechunker = Dechunker(dim=4, dropout=0.0)
        h_chunks = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0]]]
        )
        transition_probs = torch.tensor([[0.25, 0.75]])
        padding_mask = torch.tensor([[False, False, False]])

        smoothed = dechunker._apply_ema_smoothing(h_chunks, transition_probs, padding_mask)
        expected = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0],
              [0.75, 0.25, 0.0, 0.0],
              [0.1875, 0.0625, 0.75, 0.0]]]
        )

        self.assertTrue(torch.allclose(smoothed, expected, atol=1e-6))

    def test_downsampler_selects_first_token_per_chunk(self):
        downsampler = Downsampler()
        h_local = torch.tensor(
            [
                [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
                [[10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0], [50.0, 0.0]],
            ]
        )
        segment_ids = torch.tensor(
            [
                [0, 0, 1, 1, 2],
                [0, 1, 1, 2, -1],
            ],
            dtype=torch.long,
        )
        out = downsampler(h_local, segment_ids)
        expected = torch.tensor(
            [
                [[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]],
                [[10.0, 0.0], [20.0, 0.0], [40.0, 0.0]],
            ]
        )
        self.assertTrue(torch.allclose(out["chunk_repr"], expected, atol=1e-6))
        self.assertTrue(torch.equal(out["chunk_mask"], torch.tensor([[True, True, True], [True, True, True]])))

    def test_dechunker_restores_token_shape(self):
        dechunker = Dechunker(dim=4, dropout=0.0)
        h_chunks = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]
        )
        h_local = torch.zeros(1, 6, 4)
        segment_ids = torch.tensor([[0, 0, 1, 1, 2, 2]])
        boundary_probs = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]])
        src_key_padding_mask = torch.tensor([[False, False, False]])

        out = dechunker(h_chunks, h_local, segment_ids, boundary_probs, src_key_padding_mask)
        self.assertEqual(out["token_features"].shape, (1, 6, 4))
        self.assertEqual(out["upsampled"].shape, (1, 6, 4))
        self.assertTrue(torch.isfinite(out["token_features"]).all().item())

    def test_model_forward_shapes_and_padding(self):
        torch.manual_seed(7)
        model = RNADynamicModel(
            input_vocab_size=7,
            output_vocab_size=7,
            embed_dim=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            beta=0.5,
            num_classes=3,
            dropout=0.0,
        )
        model.eval()

        x = torch.randint(0, 4, (2, 8))
        # Build vectors that the new interface expects
        import numpy as np
        _bppm = torch.rand(2, 8, 8); _bppm = (_bppm + _bppm.transpose(1, 2)) / 2
        row_sum = _bppm.sum(dim=-1)
        p_safe = _bppm.clamp(min=1e-8)
        entropy = -(p_safe * p_safe.log()).sum(dim=-1) / 8.0
        S = _bppm.cumsum(dim=1).cumsum(dim=2)
        cross_pair_sum = (S[:, :, -1] - torch.diagonal(S, dim1=1, dim2=2))[:, :-1]
        mask = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )

        with torch.no_grad():
            out = model(x, row_sum=row_sum, entropy=entropy, cross_pair_sum=cross_pair_sum, mask=mask)

        self.assertEqual(out["recon_logits"].shape, (2, 8, 7))
        self.assertEqual(out["class_logits"].shape, (2, 3))
        self.assertEqual(out["segment_ids"].shape, (2, 8))
        self.assertEqual(out["chunk_repr"].shape[0], 2)
        self.assertEqual(out["token_features"].shape[:2], (2, 8))
        self.assertTrue(torch.isfinite(out["recon_logits"]).all().item())
        self.assertTrue(torch.isfinite(out["class_logits"]).all().item())
        self.assertTrue(torch.isfinite(out["boundary_probs"]).all().item())
        self.assertTrue(torch.equal(out["segment_ids"][0, -2:], torch.tensor([-1, -1])))
    def test_struct_injection_is_configurable_for_all_modes(self):
        torch.manual_seed(13)
        x = torch.randint(0, 4, (2, 8))
        _bppm = torch.zeros(2, 8, 8)
        _bppm[:, 1, 6] = 0.9
        _bppm[:, 6, 1] = 0.9
        row_sum = _bppm.sum(dim=-1)
        p_safe = _bppm.clamp(min=1e-8)
        entropy = -(p_safe * p_safe.log()).sum(dim=-1) / 8.0
        S = _bppm.cumsum(dim=1).cumsum(dim=2)
        cross_pair_sum = (S[:, :, -1] - torch.diagonal(S, dim1=1, dim2=2))[:, :-1]
        mask = torch.ones(2, 8)

        modes = [
            {"use_no_chunk": True, "use_fixed_router": False},
            {"use_no_chunk": False, "use_fixed_router": True},
            {"use_no_chunk": False, "use_fixed_router": False},
        ]
        for mode_kwargs in modes:
            with self.subTest(mode=mode_kwargs):
                model_on = RNADynamicModel(
                    input_vocab_size=7,
                    output_vocab_size=7,
                    embed_dim=16,
                    nhead=4,
                    num_layers=1,
                    dim_feedforward=32,
                    beta=0.5,
                    num_classes=3,
                    dropout=0.0,
                    use_struct_injection=True,
                    **mode_kwargs,
                )
                model_off = RNADynamicModel(
                    input_vocab_size=7,
                    output_vocab_size=7,
                    embed_dim=16,
                    nhead=4,
                    num_layers=1,
                    dim_feedforward=32,
                    beta=0.5,
                    num_classes=3,
                    dropout=0.0,
                    use_struct_injection=False,
                    **mode_kwargs,
                )
                model_on.eval()
                model_off.eval()

                with torch.no_grad():
                    out_on = model_on(x, row_sum=row_sum, entropy=entropy, cross_pair_sum=cross_pair_sum, mask=mask)
                    out_off = model_off(x, row_sum=row_sum, entropy=entropy, cross_pair_sum=cross_pair_sum, mask=mask)

                self.assertFalse(torch.allclose(out_on["h_local_struct"], out_on["h_local"], atol=1e-6))
                self.assertTrue(torch.allclose(out_off["h_local_struct"], out_off["h_local"], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
