from dataclasses import dataclass

from wald.ingest import Cell, ParsedNotebook
from wald.llm import BackendError
from wald.verifier import Verdict, verify_finding


@dataclass
class StubFinding:
    flaw_id: str
    claim_cell: int
    claim_quote: str
    code_cell: int
    code_quote: str


class StubBackend:
    provider = "stub-provider"
    model = "stub-model-1"
    kind = "api"
    gate_eligible = True

    def __init__(self, response=None, raises=None):
        self._response = response
        self._raises = raises
        self.last_system = None
        self.last_user = None

    def complete(self, system, user, schema=None):
        self.last_system = system
        self.last_user = user
        if self._raises is not None:
            raise self._raises
        return self._response


def nb() -> ParsedNotebook:
    return ParsedNotebook(
        path=None,
        cells=[
            Cell(index=0, cell_type="markdown", source="Setup notes."),
            Cell(
                index=1,
                cell_type="markdown",
                source=(
                    "LTV grows across the population.\n"
                    "This is measured on the retained cohort only."
                ),
            ),
            Cell(index=2, cell_type="code", source="imports here"),
            Cell(
                index=3,
                cell_type="code",
                source="df = df[df.status == 'active']\nltv = df.ltv.mean()",
            ),
        ],
    )


def finding() -> StubFinding:
    return StubFinding(
        flaw_id="selection-survivorship-cohort",
        claim_cell=1,
        claim_quote="LTV grows across the population.",
        code_cell=3,
        code_quote="df = df[df.status == 'active']",
    )


def test_supported_round_trip():
    backend = StubBackend({"verdict": "supported", "reason": "no scoping present"})
    verdict = verify_finding(finding(), nb(), backend)
    assert verdict == Verdict(supported=True, reason="no scoping present")


def test_unsupported_round_trip():
    backend = StubBackend({"verdict": "unsupported", "reason": "claim is scoped"})
    verdict = verify_finding(finding(), nb(), backend)
    assert verdict == Verdict(supported=False, reason="claim is scoped")


def test_prompt_contains_definition_disqualifier_and_full_cell_text():
    backend = StubBackend({"verdict": "unsupported", "reason": "x"})
    verify_finding(finding(), nb(), backend)
    prompt = backend.last_user
    assert "cohort is filtered by a condition correlated with the outcome" in prompt
    assert "explicitly scoped to the filtered group" in prompt
    assert "LTV grows across the population." in prompt
    assert "df = df[df.status == 'active']" in prompt
    # full cell text, not just the quote: the scoping sentence one line away
    assert "This is measured on the retained cohort only." in prompt
    assert "ltv = df.ltv.mean()" in prompt


def test_garbage_output_is_unsupported():
    backend = StubBackend({"nonsense": True})
    verdict = verify_finding(finding(), nb(), backend)
    assert verdict.supported is False
    assert "unparseable" in verdict.reason


def test_unknown_verdict_string_is_unsupported():
    backend = StubBackend({"verdict": "maybe", "reason": "unclear"})
    verdict = verify_finding(finding(), nb(), backend)
    assert verdict.supported is False
    assert "unparseable" in verdict.reason


def test_backend_error_is_unsupported():
    backend = StubBackend(raises=BackendError("could not parse response as JSON"))
    verdict = verify_finding(finding(), nb(), backend)
    assert verdict.supported is False
    assert "backend error" in verdict.reason
