"""Native GRU exports retain arbitrary bitwidths without AIMET dependencies."""

import json

import pytest
import torch

import gru_interface_binding as native
from quant_gru import QuantGRU

pytestmark = pytest.mark.skipif(
    not getattr(native, "cpu_only", False) and not torch.cuda.is_available(),
    reason="GRU export requires a CPU wheel or CUDA",
)


def _calibrated(tmp_path, bits, *, bidirectional=False, unsigned_gate=False):
    torch.manual_seed(39)
    device = "cpu" if getattr(native, "cpu_only", False) else "cuda"
    model = QuantGRU(3, 4, bidirectional=bidirectional).to(device).eval()
    config = {"GRU_config": {"operator_config": {
        name: {"bitwidth": bits} for name in ("weight_ih", "weight_hh")
    }}}
    if unsigned_gate:
        config["GRU_config"]["operator_config"]["update_gate_output"] = {
            "bitwidth": 4, "is_unsigned": True,
        }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    model.load_bitwidth_config(str(config_path))
    x = torch.randn(9, 2, 3, device=device) * 0.3 + 0.15
    with model.calibration_context(), torch.no_grad():
        model(x)
        model(x * 1.1)
    model.enable_pot2()
    return model, x


@pytest.mark.parametrize("bits", range(1, 33))
def test_aimet_export_preserves_all_weight_bitwidths(tmp_path, bits):
    model, _ = _calibrated(tmp_path, bits)
    document = model.export_quant_params_to_aimet_format({}, module_name="gru")
    for name in ("weight_ih", "weight_hh"):
        record = document["param_encodings"][f"gru.{name}.weight"]
        assert record["bitwidth"] == bits
        assert record["dtype"] == f"INT{bits}"


@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("unsigned_gate", [False, True])
def test_i8w4_export_and_encoding_roundtrip(tmp_path, bidirectional, unsigned_gate):
    model, x = _calibrated(tmp_path, 4, bidirectional=bidirectional, unsigned_gate=unsigned_gate)
    document = model.export_quant_params_to_aimet_format({}, module_name="gru")
    layer = document["activation_encodings"]["gru"]
    for direction in ("internal_ops", "internal_ops_reverse")[:2 if bidirectional else 1]:
        gate = layer[direction]["update_gate_output"]["output"][0]
        bits = 4 if unsigned_gate else 8
        assert (gate["dtype"], gate["bitwidth"]) == (f"UINT{bits}", bits)
    target = QuantGRU(3, 4, bidirectional=bidirectional).to(x.device).eval()
    target.load_state_dict(model.state_dict())
    assert target.load_quant_params_from_aimet_format(document, module_name="gru")
    model.use_quantization = target.use_quantization = True
    with torch.no_grad():
        for actual, expected in zip(target(x), model(x)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("bits", [4, 8, 16])
def test_native_json_format_and_encoding_roundtrip(tmp_path, bits):
    model, x = _calibrated(tmp_path, bits, bidirectional=True)
    path = tmp_path / "native.json"
    model.export_quant_params(str(path))
    document = json.loads(path.read_text())
    for direction in ("operators", "operators_reverse"):
        for record in document[direction].values():
            assert set(record) == {
                "dtype", "symmetric", "scale", "zero_point", "enc_type",
                "real_min", "real_max",
            }
    target = QuantGRU(3, 4, bidirectional=True).to(x.device).eval()
    target.load_state_dict(model.state_dict())
    target.load_quant_params(str(path))
    exported = target.export_quant_params_to_aimet_format({}, module_name="gru")
    for name in ("weight_ih", "weight_hh"):
        assert exported["param_encodings"][f"gru.{name}.weight"]["bitwidth"] == bits
    model.use_quantization = target.use_quantization = True
    with torch.no_grad():
        for actual, expected in zip(target(x), model(x)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
