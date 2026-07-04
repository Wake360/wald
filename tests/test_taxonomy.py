from wald.taxonomy import load_fusion_rules, load_taxonomy


def test_taxonomy_loads_all_classes():
    taxonomy = load_taxonomy()
    assert "selection-survivorship-cohort" in taxonomy
    assert "significance-meaningless" in taxonomy
    assert "regression-to-mean-claim" in taxonomy


def test_exactly_three_classes_narrative_enabled():
    taxonomy = load_taxonomy()
    enabled = {fid for fid, d in taxonomy.items() if d.narrative_enabled}
    assert enabled == {
        "selection-survivorship-cohort",
        "significance-meaningless",
        "regression-to-mean-claim",
    }


def test_narrative_enabled_classes_have_disqualifiers():
    taxonomy = load_taxonomy()
    for fid in (
        "selection-survivorship-cohort",
        "significance-meaningless",
        "regression-to-mean-claim",
    ):
        assert 2 <= len(taxonomy[fid].disqualifiers) <= 4


def test_fusion_rules_load():
    rules = load_fusion_rules()
    assert {r.id for r in rules} == {
        "survivorship-fused",
        "pre-cv-fused",
        "solo-narrative",
    }


def test_fusion_rule_flaw_ids_exist_in_taxonomy():
    taxonomy = load_taxonomy()
    rules = load_fusion_rules()
    for r in rules:
        if r.static is not None:
            assert r.static.flaw_id in taxonomy
        if r.narrative.flaw_id:
            assert r.narrative.flaw_id in taxonomy


def test_solo_rule_has_no_static_match():
    rules = {r.id: r for r in load_fusion_rules()}
    assert rules["solo-narrative"].static is None
    assert rules["solo-narrative"].narrative.any_enabled_finding is True


def test_fusion_rule_confidences():
    rules = {r.id: r for r in load_fusion_rules()}
    assert rules["survivorship-fused"].emit_confidence == 0.91
    assert rules["pre-cv-fused"].emit_confidence == 0.88
    assert rules["solo-narrative"].emit_confidence == 0.80
