from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_primary_config_is_miso_and_uses_documented_absolute_phase_mapping():
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text())["env"]
    assert int(cfg["num_bs_antennas"]) >= 2
    assert cfg["phase_action_mode"] == "absolute"


def test_historical_outputs_are_marked_nonfinal():
    paper = (ROOT / "paper_results/README.md").read_text().lower()
    rerun = (ROOT / "latex_thesis/MISO_RERUN_NOTE.md").read_text().lower()
    assert "historical" in paper
    assert "do not submit" in rerun
