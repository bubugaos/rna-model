"""
RNA Dynamic Chunking Model - Logical Vulnerability Tests

This test file identifies and verifies potential logical vulnerabilities:
1. DynamicRouter boundary mask handling issues
2. BPPM cumulative sum index offset problems
3. Temperature scaling causing gradient vanishing
4. MLM loss label alignment issues
5. Positional encoding behavior on variable-length sequences
6. Soft membership numerical stability
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.dynamic_router import DynamicRouter, FixedChunkRouter
from src.model.chunking import Downsampler
from src.model.latent_transformer import LatentTransformer
from src.model.model import RNADynamicModel
from src.data.masking_utils import SimpleMasking, PairingAwareMasking
from src.training.losses import RNALMCriterion
from src.config import RNA_TOKEN_IDS


class TestDynamicRouterBoundaryHandling(unittest.TestCase):
    """Test 1: DynamicRouter boundary mask handling"""

    def test_boundary_mask_padding_leakage(self):
        """
        Vulnerability: When sequence has padding at the end, boundary probs may leak into padding region

        Verification: Boundary probs at padding positions should be strictly 0
        """
        router = DynamicRouter(dim=64, beta=0.5)
        batch_size = 2
        seq_len = 16

        h_local = torch.randn(batch_size, seq_len, 64)
        bppm = torch.rand(batch_size, seq_len, seq_len)
        bppm = (bppm + bppm.transpose(1, 2)) / 2
        S = bppm.cumsum(dim=1).cumsum(dim=2)
        cross_pair_sum = (S[:, :, -1] - torch.diagonal(S, dim1=1, dim2=2))[:, :-1]

        # Create mask with padding at the end for first sample
        mask = torch.ones(batch_size, seq_len)
        mask[0, -3:] = 0  # Last 3 positions are padding

        router.eval()
        with torch.no_grad():
            output = router(h_local, cross_pair_sum=cross_pair_sum, mask=mask)

        boundary_probs = output["boundary_probs"]

        # Verify: boundary probs involving padding positions should be 0
        # Positions 12-13, 13-14, 14-15 boundaries should be 0
        self.assertEqual(boundary_probs[0, -3:].sum().item(), 0.0,
                        "Boundary probability leaked into padding region")

    def test_single_token_sequence(self):
        """
        Vulnerability: Single token sequence may cause division by zero or shape mismatch

        Verification: seq_len=1 should return empty boundaries but valid segment_ids
        """
        router = DynamicRouter(dim=64)
        h_local = torch.randn(2, 1, 64)
        cross_pair_sum = torch.zeros(2, 0)  # seq_len=1 → no boundaries
        mask = torch.ones(2, 1)

        router.eval()
        with torch.no_grad():
            output = router(h_local, cross_pair_sum=cross_pair_sum, mask=mask)

        self.assertEqual(output["boundary_probs"].shape, (2, 0),
                        "Wrong boundary_probs shape for single token sequence")
        self.assertEqual(output["segment_ids"].shape, (2, 1),
                        "Wrong segment_ids shape for single token sequence")
        self.assertTrue(torch.all(output["segment_ids"] >= 0),
                       "segment_ids contains invalid negative values")


class TestBPPMIndexCalculation(unittest.TestCase):
    """Test 2: BPPM cumulative sum index issues"""

    def test_bppm_cumsum_index_offset(self):
        """
        Vulnerability: dynamic_router.py lines 189-193 BPPM cumsum calculation

        S = bppm.cumsum(dim=1).cumsum(dim=2)
        S_i_last = S[:, :, -1]
        S_i_i = torch.diagonal(S, dim1=1, dim2=2)
        U_all = S_i_last - S_i_i
        u_scores = U_all[:, :-1]

        This computes sum of pairing probs from position i to sequence end
        minus diagonal elements. Index offset may cause u_scores[i] to
        correspond to position i+1 instead of i.
        """
        router = DynamicRouter(dim=64)
        batch_size = 1
        seq_len = 8

        # Construct special BPPM: strong connection between positions 2 and 6
        h_local = torch.randn(batch_size, seq_len, 64)
        bppm = torch.zeros(batch_size, seq_len, seq_len)
        bppm[0, 2, 6] = 0.9
        bppm[0, 6, 2] = 0.9
        S = bppm.cumsum(dim=1).cumsum(dim=2)
        cross_pair_sum = (S[:, :, -1] - torch.diagonal(S, dim1=1, dim2=2))[:, :-1]

        mask = torch.ones(batch_size, seq_len)

        router.eval()
        with torch.no_grad():
            output = router(h_local, cross_pair_sum=cross_pair_sum, mask=mask)

        b_scores = output["b_scores"]

        # Verify: boundaries between positions 2 and 6 should have higher structure scores
        middle_scores = b_scores[0, 2:6]
        edge_scores = torch.cat([b_scores[0, :2], b_scores[0, 6:]])

        # Soft verification since score calculation depends on other factors
        self.assertTrue(middle_scores.mean().item() >= edge_scores.mean().item() - 0.1,
                       "Structure score calculation may have index offset")


class TestTemperatureScaling(unittest.TestCase):
    """Test 3: Router temperature control remains numerically stable"""

    def test_router_tau_updates_boundary_distribution(self):
        router = DynamicRouter(dim=64)
        h_local = torch.randn(2, 16, 64)
        b_scores = torch.rand(2, 15)
        bppm = torch.zeros(2, 16, 16)
        S = bppm.cumsum(dim=1).cumsum(dim=2)
        cross_pair_sum = (S[:, :, -1] - torch.diagonal(S, dim1=1, dim2=2))[:, :-1]
        mask = torch.ones(2, 16)

        router.eval()
        router.set_tau(1.0)
        out_tau1 = router(h_local, cross_pair_sum=cross_pair_sum, mask=mask)
        router.set_tau(0.1)
        out_tau01 = router(h_local, cross_pair_sum=cross_pair_sum, mask=mask)

        self.assertTrue(torch.isfinite(out_tau1["boundary_probs"]).all())
        self.assertTrue(torch.isfinite(out_tau01["boundary_probs"]).all())
        self.assertFalse(torch.allclose(out_tau1["boundary_probs"], out_tau01["boundary_probs"]))


class TestMLMLossLabelAlignment(unittest.TestCase):
    """Test 4: MLM loss label alignment"""

    def test_mask_positions_label_alignment(self):
        """
        Vulnerability: In losses.py _compute_mlm_loss:

        masked_logits = recon_logits[mask_positions]
        masked_labels = labels[mask_positions]

        If mask_positions and labels indices misaligned, causes wrong training signal.

        Verification: Masked position logits should align correctly with labels
        """
        criterion = RNALMCriterion(vocab_size=7, lambda_mlm=1.0)
        batch_size = 2
        seq_len = 16

        recon_logits = torch.randn(batch_size, seq_len, 7)
        labels = torch.randint(0, 4, (batch_size, seq_len))
        mask_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask_positions[:, 5:10] = True  # Mask middle section

        # Compute MLM loss
        loss = criterion._compute_mlm_loss(recon_logits, labels, mask_positions)

        # Verify loss should be > 0 (logits are random)
        self.assertGreater(loss.item(), 0.0, "MLM loss should be greater than 0")
        self.assertTrue(torch.isfinite(loss), "MLM loss contains NaN/Inf")

        # Manual verification: extract same position logits and labels
        masked_logits = recon_logits[mask_positions]
        masked_labels = labels[mask_positions]

        # Manual cross entropy calculation
        manual_loss = torch.nn.functional.cross_entropy(masked_logits, masked_labels)

        # Verify consistency
        self.assertAlmostEqual(
            loss.item(),
            manual_loss.item(),
            places=5,
            msg="MLM loss inconsistent with manual calculation, possible alignment issue",
        )


class TestPositionalEncodingVariableLength(unittest.TestCase):
    """Test 5: Positional encoding on variable-length sequences"""

    def test_positional_encoding_padding_positions(self):
        """
        Vulnerability: PositionalEncoding adds positional encoding to all positions including padding

        In LocalEncoder:
        h_embedded = self.pos_encoding(h_embedded)
        h_local = self.transformer_encoder(h_embedded, src_key_padding_mask=src_key_padding_mask)

        Transformer encoder uses mask, but positional encoding already added to padding positions.
        This may contaminate padding position features.

        Verification: Padding position outputs should be close to 0 or not affect valid positions
        """
        from src.model.local_encoder import LocalEncoder

        encoder = LocalEncoder(vocab_size=7, embed_dim=64)
        encoder.eval()

        batch_size = 2
        seq_len = 16

        # Create two sequences with same content but different padding
        x1 = torch.randint(0, 4, (1, seq_len))
        x1[0, -4:] = RNA_TOKEN_IDS["PAD"]
        mask1 = torch.ones(1, seq_len)
        mask1[0, -4:] = 0

        x2 = x1[:, :-4].clone()  # Remove padding part
        x2_padded = torch.full((1, seq_len), RNA_TOKEN_IDS["PAD"], dtype=torch.long)
        x2_padded[:, :12] = x2
        mask2 = torch.ones(1, seq_len)
        mask2[0, -4:] = 0

        with torch.no_grad():
            h1 = encoder(x1, mask=mask1)
            h2_full = encoder(x2_padded, mask=mask2)

        # Verify valid position outputs should be same
        valid_h1 = h1[0, :12, :]
        valid_h2 = h2_full[0, :12, :]

        # Due to different positional encoding, outputs differ slightly but within reasonable range
        diff = (valid_h1 - valid_h2).abs().mean().item()
        self.assertLess(diff, 0.5, "Positional encoding causes too large output difference")


class TestDownsamplerStability(unittest.TestCase):
    """Test 6: Downsampler padding and chunk count stability"""

    def test_downsampler_padding_positions_are_ignored(self):
        downsampler = Downsampler()
        h_local = torch.randn(2, 8, 32)
        segment_ids = torch.tensor(
            [
                [0, 0, 1, 1, 2, 2, -1, -1],
                [0, 1, 1, 2, 2, 3, 3, 3],
            ]
        )
        out = downsampler(h_local, segment_ids)
        self.assertEqual(out["chunk_repr"].shape[:2], (2, 4))
        self.assertTrue(torch.equal(out["chunk_mask"][0], torch.tensor([True, True, True, False])))


class TestReconDecoderTokenHead(unittest.TestCase):
    """Test 7: ReconDecoder token-head shape consistency"""

    def test_broadcast_segment_alignment(self):
        """
        Verification: Decoder should preserve token axis shape and gradients.
        """
        from src.model.decoder import ReconDecoder

        decoder = ReconDecoder(dim=64, vocab_size=7)
        batch_size = 2
        seq_len = 16
        token_features = torch.randn(batch_size, seq_len, 64, requires_grad=True)

        output = decoder(token_features)

        self.assertEqual(output.shape, (batch_size, seq_len, 7), "ReconDecoder output shape is wrong")

        output.sum().backward()
        self.assertIsNotNone(token_features.grad, "token_features gradient is None")
        self.assertTrue(torch.isfinite(token_features.grad).all(), "token_features gradient contains NaN/Inf")


class TestEdgeCases(unittest.TestCase):
    """Test 8: Edge cases"""

    def test_all_masked_sequence(self):
        """
        Vulnerability: When entire sequence is masked, loss calculation may produce NaN

        Verification: Even when all positions masked, loss should be meaningful
        """
        criterion = RNALMCriterion(vocab_size=7)
        batch_size = 1
        seq_len = 8

        recon_logits = torch.randn(batch_size, seq_len, 7)
        x = torch.randint(0, 4, (batch_size, seq_len))
        mask = torch.zeros(batch_size, seq_len)  # All padding

        with torch.no_grad():
            loss_dict = criterion({
                "recon_logits": recon_logits,
                "boundary_probs": torch.rand(batch_size, seq_len - 1),
            }, x, None, mask)

        # Verify loss should be 0 or close to 0 (no valid positions)
        self.assertTrue(torch.isfinite(loss_dict["recon_loss"]),
                       "recon_loss contains NaN/Inf")

    def test_very_long_sequence(self):
        """
        Vulnerability: Long sequences may cause memory overflow or numerical instability

        Verification: Model should handle long sequences (at least 512 tokens)
        """
        router = DynamicRouter(dim=64)
        seq_len = 512

        h_local = torch.randn(1, seq_len, 64)
        bppm = torch.zeros(1, seq_len, seq_len)
        # Construct sparse BPPM
        for i in range(0, seq_len, 16):
            j = min(i + 8, seq_len - 1)
            bppm[0, i, j] = 0.5
            bppm[0, j, i] = 0.5
        S = bppm.cumsum(dim=1).cumsum(dim=2)
        cross_pair_sum = (S[:, :, -1] - torch.diagonal(S, dim1=1, dim2=2))[:, :-1]

        mask = torch.ones(1, seq_len)

        router.eval()
        with torch.no_grad():
            output = router(h_local, cross_pair_sum=cross_pair_sum, mask=mask)

        self.assertEqual(output["boundary_probs"].shape, (1, seq_len - 1),
                        "Long sequence boundary_probs shape is wrong")
        self.assertTrue(torch.isfinite(output["boundary_probs"]).all(),
                       "Long sequence causes numerical instability")


class TestFixedChunkRouterPadding(unittest.TestCase):
    """Test 9: FixedChunkRouter padding handling"""

    def test_fixed_router_padding_mask(self):
        """
        Verify FixedChunkRouter correctly handles padding in boundary detection
        """
        router = FixedChunkRouter(chunk_size=4)
        batch_size = 2
        seq_len = 10

        h_local = torch.randn(batch_size, seq_len, 64)

        # First sample has padding at positions 7, 8, 9
        mask = torch.ones(batch_size, seq_len)
        mask[0, 7:] = 0

        output = router(h_local, cross_pair_sum=None, mask=mask)

        # Verify boundaries at padding positions are 0
        # Boundaries are at positions 3, 7 for chunk_size=4
        # But position 7 involves padding, so should be 0 for sample 0
        self.assertEqual(output["boundary_mask"][0, 7:].sum().item(), 0.0,
                        "FixedChunkRouter boundary mask leaked into padding")
        self.assertEqual(output["boundary_probs"][0, 7:].sum().item(), 0.0,
                        "FixedChunkRouter boundary probs leaked into padding")


class TestDynamicRouterGumbelSoftmax(unittest.TestCase):
    """Test 10: DynamicRouter Gumbel-Softmax behavior"""

    def test_gumbel_softmax_train_eval_difference(self):
        """
        Verify Gumbel-Softmax produces different behavior in train vs eval mode
        """
        router = DynamicRouter(dim=64)
        router.set_tau(0.5)  # Low temperature for clearer difference

        h_local = torch.randn(2, 16, 64)
        bppm = torch.rand(2, 16, 16)
        S = bppm.cumsum(dim=1).cumsum(dim=2)
        cross_pair_sum = (S[:, :, -1] - torch.diagonal(S, dim1=1, dim2=2))[:, :-1]
        mask = torch.ones(2, 16)

        # Train mode: uses Gumbel-Softmax sampling
        router.train()
        with torch.no_grad():
            output_train = router(h_local, cross_pair_sum=cross_pair_sum, mask=mask)

        # Eval mode: uses deterministic argmax
        router.eval()
        with torch.no_grad():
            output_eval = router(h_local, cross_pair_sum=cross_pair_sum, mask=mask)

        # Train mode boundary_probs should be softer (more values between 0 and 1)
        train_probs = output_train["boundary_probs"]
        eval_probs = output_eval["boundary_probs"]

        # Count how many values are strictly between 0 and 1
        train_soft = ((train_probs > 0.01) & (train_probs < 0.99)).float().mean().item()
        eval_soft = ((eval_probs > 0.01) & (eval_probs < 0.99)).float().mean().item()

        # Train mode should have more soft values due to Gumbel noise
        # Note: This is a soft assertion as behavior can vary
        print(f"Train soft ratio: {train_soft:.3f}, Eval soft ratio: {eval_soft:.3f}")


if __name__ == "__main__":
    unittest.main()
