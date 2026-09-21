"""End-to-end GRU training comparison on Speech Commands v0.02."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import torch

from speech_commands_gru_training import ExperimentConfig, run_training_comparison


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "tests/results/speech_commands_gru_training.json"


class SpeechCommandsGruTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset_root = os.environ.get("QUANT_GRU_SPEECH_COMMANDS_ROOT")
        if not dataset_root:
            raise unittest.SkipTest(
                "set QUANT_GRU_SPEECH_COMMANDS_ROOT to run the real-data test"
            )
        cls.dataset_root = Path(dataset_root)
        if not cls.dataset_root.is_dir():
            raise RuntimeError(f"Speech Commands dataset not found: {cls.dataset_root}")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("the QuantGRU training comparison requires CUDA")

    def test_replacing_only_gru_preserves_real_training_quality(self) -> None:
        report = run_training_comparison(
            ExperimentConfig(
                dataset_root=self.dataset_root,
                labels=("yes", "no", "up", "down"),
                train_samples_per_label=128,
                validation_samples_per_label=32,
                test_samples_per_label=32,
                hidden_size=64,
                batch_size=32,
                epochs=10,
                learning_rate=3.0e-3,
                calibration_batches=4,
                calibration_refresh_epochs=1,
                quant_bitwidths=(8, 16),
                seed=20260921,
            )
        )
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        baseline = report["training"]["torch_gru"]
        native_float = report["training"]["quant_gru_float"]
        quantized = {
            bitwidth: report["training"][f"quant_gru_qat_{bitwidth}bit"]
            for bitwidth in (8, 16)
        }

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["replacement"]["changed_module"], "gru")
        self.assertEqual(report["quantization"]["bitwidths"], [8, 16])
        self.assertEqual(
            report["replacement"]["initial_shared_state_max_abs_diff"],
            {
                "quant_gru_float": 0.0,
                "quant_gru_qat_8bit": 0.0,
                "quant_gru_qat_16bit": 0.0,
            },
        )

        for result in (baseline, native_float, *quantized.values()):
            self.assertLess(result["final_train_loss"], result["initial_train_loss"])
            self.assertGreater(result["parameter_update_norm"], 0.0)

        self.assertGreaterEqual(baseline["best_validation_accuracy"], 0.70)
        self.assertGreaterEqual(baseline["final_test_accuracy"], 0.80)
        self.assertGreaterEqual(native_float["best_validation_accuracy"], 0.70)
        self.assertGreaterEqual(native_float["final_test_accuracy"], 0.80)
        self.assertGreaterEqual(
            native_float["best_validation_accuracy"],
            baseline["best_validation_accuracy"] - 0.01,
        )
        self.assertGreaterEqual(
            native_float["final_test_accuracy"],
            baseline["final_test_accuracy"] - 0.01,
        )

        for bitwidth, result in quantized.items():
            with self.subTest(bitwidth=bitwidth):
                calibration = report["quantization"]["calibration"][
                    f"quant_gru_qat_{bitwidth}bit"
                ]
                self.assertEqual(calibration["selection"], "balanced_round_robin")
                self.assertEqual(calibration["sample_count"], 128)
                self.assertEqual(calibration["label_counts"], [32, 32, 32, 32])
                self.assertEqual(len(result["calibration_refreshes"]), 10)
                self.assertTrue(
                    all(
                        refresh["label_counts"] == [32, 32, 32, 32]
                        for refresh in result["calibration_refreshes"]
                    )
                )
                self.assertGreaterEqual(result["best_validation_accuracy"], 0.70)
                self.assertGreaterEqual(result["final_test_accuracy"], 0.80)
                self.assertGreaterEqual(
                    result["best_validation_accuracy"],
                    baseline["best_validation_accuracy"] - 0.05,
                )

        error_8 = quantized[8]["final_quantization_error"]
        error_16 = quantized[16]["final_quantization_error"]
        self.assertLess(error_8["mae"], 0.03)
        self.assertLess(error_8["normalized_mae"], 0.015)
        self.assertGreaterEqual(error_8["cosine"], 0.9998)
        self.assertLess(error_16["mae"], 0.001)
        self.assertLess(error_16["normalized_mae"], 0.001)
        self.assertLess(error_16["mae"], error_8["mae"] * 0.05)
        self.assertLess(
            error_16["normalized_mae"],
            error_8["normalized_mae"] * 0.05,
        )
        self.assertGreaterEqual(error_16["cosine"], 0.999999)
        self.assertGreaterEqual(error_8["prediction_agreement"], 0.98)
        self.assertEqual(error_16["prediction_agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()
