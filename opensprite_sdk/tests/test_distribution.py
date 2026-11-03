"""Distribution rules must exclude local calibration, including relocation."""
import importlib.util
from pathlib import Path

import pytest


ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("sdk_distribution_check",ROOT/"tools/check_standalone.py")
CHECK=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


@pytest.mark.parametrize("name",[
    "config/hardware.local.json", "config/review-check.local.json",
    "some-prefix/config/hardware.local.json", ".reference/repo/README.md",
    "outputs/extraction/calibration.json", ".venv/lib/module.py",
])
def test_private_inputs_never_enter_relocated_distribution(name):
    assert CHECK.excluded(Path(name))


@pytest.mark.parametrize("name",[
    "config/hardware.jetson.example.json", ".gitignore", ".gitattributes",
    "src/opensprite_sdk/profiles/legacy_v4.json", "native/vendor/LICENSE",
])
def test_public_templates_and_repository_rules_are_distributable(name):
    assert not CHECK.excluded(Path(name))
