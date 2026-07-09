import re
from pathlib import Path

import yaml

REPO = Path(__file__).parent.parent


def test_action_version_matches_package():
    # pyproject.toml's version is a plain `version = "x.y.z"` line; a regex
    # keeps this test dependency-free rather than requiring tomllib (py3.11+)
    # or a tomli backport for the py3.10 CI leg.
    pyproject = (REPO / "pyproject.toml").read_text()
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject)
    assert match, "no version found in pyproject.toml"
    pkg_version = match.group(1)

    action = yaml.safe_load((REPO / "action.yml").read_text())
    action_version = action["inputs"]["version"]["default"]

    assert action_version == pkg_version
