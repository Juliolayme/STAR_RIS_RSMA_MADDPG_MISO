from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_primary_config_uses_structured_physics_informed_action():
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text())["env"]
    assert int(cfg["num_bs_antennas"]) >= 2
    assert cfg["bs_action_mode"] == "structured_rzf"
    assert cfg["phase_action_mode"] == "residual"
    assert 0.0 < float(cfg["bs_rzf_regularization"]) <= 1.0
    assert 0.0 <= float(cfg["bs_rzf_mix_prior"]) <= 1.0


def test_historical_outputs_are_marked_nonfinal():
    paper = (ROOT / "paper_results/README.md").read_text().lower()
    rerun = (ROOT / "latex_thesis/MISO_RERUN_NOTE.md").read_text().lower()
    assert "historical" in paper
    assert "do not submit" in rerun
