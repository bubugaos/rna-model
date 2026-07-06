import os
import pickle
import tempfile
import unittest

import numpy as np
import torch

from src.config import ExperimentConfig, validate_config
from src.data.data_loader import RealRNADataset
from src.data.masking_utils import PairingAwareMasking


def build_pair_bppm(length: int, left: int, right: int, prob: float) -> np.ndarray:
    bppm = np.zeros((length, length), dtype=np.float32)
    bppm[left, right] = prob
    bppm[right, left] = prob
    return bppm


class TestPairingAwareMasking(unittest.TestCase):
    def test_pairing_aware_requires_bppm(self):
        masker = PairingAwareMasking()
        x = torch.randint(0, 4, (1, 80))
        with self.assertRaisesRegex(ValueError, "top1_partner"):
            masker(x, None)

    def test_pairing_aware_masks_partner_when_coupling_enabled(self):
        x = torch.randint(0, 4, (1, 80))
        bppm = torch.from_numpy(build_pair_bppm(80, 10, 60, 0.95)).unsqueeze(0)
        # Derive top1_partner and top1_prob from bppm
        bppm_no_diag = bppm.clone()
        bppm_no_diag.diagonal(dim1=1, dim2=2).fill_(0)
        top1_prob, top1_partner = bppm_no_diag.max(dim=-1)
        masker = PairingAwareMasking(
            mask_prob=0.2,
            coupled_prob=1.0,
            pairing_threshold=0.3,
            replace_prob=1.0,
            random_prob=0.0,
        )

        selected_count = 0
        coupled_count = 0
        for seed in range(200):
            torch.manual_seed(seed)
            _, mask_positions, _ = masker(x, top1_partner, top1_prob)
            if mask_positions[0, 10]:
                selected_count += 1
                if mask_positions[0, 60]:
                    coupled_count += 1

        self.assertGreater(selected_count, 0)
        self.assertEqual(coupled_count, selected_count)

    def test_validate_config_accepts_pairing_aware_only(self):
        config = ExperimentConfig()
        config.masking.masking_type = "pairing_aware"
        self.assertEqual(validate_config(config), [])

        config.masking.masking_type = "structure_aware"
        errors = validate_config(config)
        self.assertTrue(any("Invalid masking_type" in error for error in errors))

    def test_data_loader_initializes_pairing_aware_masker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = os.path.join(tmpdir, "sample.fasta")
            pkl_path = os.path.join(tmpdir, "sample.pkl")
            seq = "AUGC" * 20

            with open(fasta_path, "w", encoding="utf-8") as fasta_file:
                fasta_file.write(">seq0 family0\n")
                fasta_file.write(seq + "\n")

            with open(pkl_path, "wb") as pkl_file:
                pickle.dump([build_pair_bppm(len(seq), 10, 60, 0.9)], pkl_file)

            dataset = RealRNADataset(
                fasta_path=fasta_path,
                pkl_path=pkl_path,
                use_masking=True,
                masking_type="pairing_aware",
                masking_config={
                    "mask_prob": 0.2,
                    "coupled_prob": 1.0,
                    "pairing_threshold": 0.3,
                    "replace_prob": 1.0,
                    "random_prob": 0.0,
                },
            )

            self.assertEqual(dataset.masker.__class__.__name__, "PairingAwareMasking")

            torch.manual_seed(0)
            result = dataset[0]
            masked_x, raw_x, seq_len, mask_pos, labels, family_label, dotbracket, row_sum, entropy, cross_pair = result

            self.assertEqual(seq_len, len(seq))
            self.assertEqual(masked_x.shape, raw_x.shape)
            self.assertEqual(mask_pos.dtype, torch.bool)
            self.assertEqual(labels.shape, raw_x.shape)
            self.assertEqual(row_sum.shape, (len(seq),))
            self.assertEqual(entropy.shape, (len(seq),))
            self.assertEqual(cross_pair.shape, (len(seq) - 1,))
            self.assertIsInstance(family_label, int)
            self.assertIsNone(dotbracket)


if __name__ == "__main__":
    unittest.main()
