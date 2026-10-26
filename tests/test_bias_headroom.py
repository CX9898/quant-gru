"""Fixed GRU bias grids permit small QAT updates, including zero channels."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from quant_gru import QuantGRU


def export(model, path):
    model.export_quant_params(str(path))
    return json.loads(path.read_text())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class GruBiasHeadroomTest(unittest.TestCase):
    def test_updates_preserve_fixed_encoding_and_nonzero_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encodings.json"
            for bits in (8, 16):
                for pot2 in (False, True):
                    for initial, step in ((0.0, 1e-4), (0.2, 0.05), (-0.2, -0.05)):
                        with self.subTest(bits=bits, pot2=pot2, initial=initial):
                            torch.manual_seed(42)
                            model = QuantGRU(4, 4).cuda()
                            model.set_all_bitwidth(bits)
                            if pot2:
                                model.enable_pot2()
                            with torch.no_grad():
                                model.bias_ih_l0.fill_(initial)
                                model.bias_hh_l0.fill_(initial)
                            x = torch.randn(5, 2, 4, device="cuda") * 0.5
                            model.calibrating = True
                            with torch.no_grad():
                                model(x)
                            model.calibrating = False
                            model.finalize_calibration()
                            model.use_quantization = True
                            before = export(model, path)
                            model.set_quant_params_locked(True)
                            scales = torch.as_tensor(before["operators"]["bias_ih"]["scale"], device="cuda")
                            codes = torch.round(torch.tensor(initial + step, device="cuda") / scales)
                            self.assertTrue(torch.all(codes <= 2 ** (bits - 1) - 1))
                            self.assertTrue(torch.all(codes >= -(2 ** (bits - 1))))
                            if initial == 0:
                                self.assertTrue(torch.all(codes != 0))
                            with torch.no_grad():
                                model.bias_ih_l0.add_(step)
                                model.bias_hh_l0.add_(step)
                            model.train()
                            output, _ = model(x)
                            output.square().mean().backward()
                            self.assertGreater(model.bias_ih_l0.grad.abs().max().item(), 0.0)
                            self.assertEqual(before, export(model, path))
                            clone = QuantGRU(4, 4).cuda()
                            clone.load_state_dict(model.state_dict())
                            path.write_text(json.dumps(before))
                            clone.load_quant_params(str(path))
                            clone.use_quantization = True
                            model.eval()
                            clone.eval()
                            with torch.no_grad():
                                torch.testing.assert_close(clone(x)[0], model(x)[0], rtol=0, atol=0)

    def test_zero_bias_all_calibration_methods_and_granularities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encodings.json"
            for method in ("minmax", "percentile", "sqnr"):
                for granularity in range(3):
                    with self.subTest(method=method, granularity=granularity):
                        torch.manual_seed(42)
                        model = QuantGRU(4, 4).cuda()
                        model.calibration_method = method
                        model._bitwidth_config.bw_granularity_ = granularity
                        model._bitwidth_config.br_granularity_ = granularity
                        with torch.no_grad():
                            model.bias_ih_l0.zero_()
                            model.bias_hh_l0.zero_()
                            model.calibrating = True
                            model(torch.randn(5, 2, 4, device="cuda") * 0.5)
                        model.calibrating = False
                        model.finalize_calibration()
                        encodings = export(model, path)
                        for name in ("bias_ih", "bias_hh"):
                            scales = torch.as_tensor(encodings["operators"][name]["scale"])
                            codes = torch.round(1e-4 / scales)
                            self.assertTrue(torch.all(codes > 0))
                            self.assertTrue(torch.all(codes <= 127))


if __name__ == "__main__":
    unittest.main()
