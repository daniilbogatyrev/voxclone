"""GPU-free tests for the one notebook builder (voice_cloning_study.ipynb).

This notebook is the full deliverable: it documents the pipeline, RE-SCORES the committed
results with the real eval code, charts them, explains the 90% number, lets the reader clone
their OWN voice (few-shot) and documents fine-tuning. It builds for two audiences:

* ``public`` (module default) — the version committed to the public repo: it carries NO author
  audio and NO fine-tuned-weight key (``f5_finetuned``); the "hear my voice" section is omitted.
* ``professor`` — the ZIP version that keeps the "hear my cloned voice" section.

The tests assert those pillars are present per audience and that every cell is structurally
sound — without a GPU.
"""
import nbformat
import pytest

from notebooks import build_voice_cloning_study as B


def _src(audience="public"):
    return "\n".join(c.source for c in B.build_notebook(audience).cells)


@pytest.mark.parametrize("audience", B.AUDIENCES)
def test_builds_valid_notebook_pinned_to_voxclone_kernel(audience, tmp_path):
    p = tmp_path / "voice_cloning_study.ipynb"
    B.write_notebook(str(p), audience=audience)
    loaded = nbformat.read(str(p), as_version=4)
    nbformat.validate(loaded)                       # raises if invalid
    assert loaded.metadata["kernelspec"]["name"] == "voxclone"
    assert {c.cell_type for c in loaded.cells} == {"markdown", "code"}


@pytest.mark.parametrize("audience", B.AUDIENCES)
def test_documents_results_with_real_scoring_code(audience):
    src = _src(audience)
    # Re-derives scores from the committed results with the REAL formula (not a paraphrase).
    assert "ab_eval_finetune_results.json" in src
    assert "from voxclone.eval.picker import combined_score" in src
    assert "combined_score(" in src
    assert "inspect.getsource" in src           # shows the real source in-notebook


@pytest.mark.parametrize("audience", B.AUDIENCES)
def test_explains_the_similarity_method(audience):
    src = _src(audience)
    assert "voxclone.eval import similarity" in src
    assert "reference_ceiling" in src           # the real-vs-real ceiling derivation
    assert 'res["ceiling"]' in src              # ceiling used in the leaderboard re-derivation


def test_professor_keeps_the_ninety_percent():
    # The author's measured headline number stays in the ZIP build (it's HIS voice).
    src = _src("professor")
    assert "≈ 90" in src
    assert "90{,}1" in src                       # the 90.1 % derivation (LaTeX form)
    assert "Schlüsselzahl" in src


def test_public_drops_the_fixed_percentage():
    # A public reader clones their OWN voice — no fixed "= 90 %" result is promised.
    src = _src("public")
    assert "≈ 90" not in src
    assert "90 %" not in src
    assert "90{,}1" not in src
    # ...but the measurement METHOD is still taught, framed as "measure your own".
    assert "Wie Ähnlichkeit gemessen wird" in src
    assert "keine feste Zahl" in src
    assert "voxclone-eval" in src


def test_public_clones_your_own_voice_without_author_audio():
    src = _src("public")
    # Few-shot: clone YOUR OWN voice — upload + custom reference (zero-shot only).
    assert "FileUpload" in src
    assert "ref_path=" in src                   # custom reference threaded into say()
    assert "ref_text=" in src
    assert "f5_zeroshot" in src                 # best zero-shot engine offered
    # The public build must NOT carry the author's fine-tuned-weight voice or its audio.
    assert "f5_finetuned" not in src
    assert "meine geklonte Stimme" not in src


def test_professor_build_keeps_hear_my_voice():
    src = _src("professor")
    # The ZIP version keeps the author's voice so it can embed audio on the GPU box.
    assert "f5_finetuned" in src
    assert "studio.say(" in src
    assert "meine geklonte Stimme" in src
    # ...and still offers clone-your-own.
    assert "FileUpload" in src and "f5_zeroshot" in src


@pytest.mark.parametrize("audience", B.AUDIENCES)
def test_guides_fine_tuning_your_own_voice(audience):
    src = _src(audience)
    # Links the read-aloud script + the runbook, and surfaces the few-shot sentence.
    assert "data/recording_script.md" in src
    assert "RUNBOOK_finetune.md" in src
    assert "voxclone-train" in src
    assert "This is my natural speaking voice" in src   # the few-shot / calibration sentence


@pytest.mark.parametrize("audience", B.AUDIENCES)
def test_never_trains_and_stays_consent_aware(audience):
    src = _src(audience)
    assert ".train(" not in src                 # documents training, never runs it
    assert "Einwilligung" in src                # consent is stated for clone-your-own


def test_audio_cells_degrade_gracefully():
    # The live-audio cells must never abort the notebook when the GPU/env is absent.
    say_cells = [c.source for c in B.cells
                 if c.cell_type == "code" and "studio.say(" in c.source]
    assert say_cells, "expected at least one studio.say cell"
    for cell in say_cells:
        assert "CAN_GENERATE" in cell           # gated on capability
        assert "try:" in cell and "except Exception" in cell


def test_preflight_sets_can_generate_and_force_nogen(monkeypatch):
    guard = next(c.source for c in B.cells
                 if c.cell_type == "code" and "CAN_GENERATE" in c.source and "FORCE_NOGEN" in c.source)
    # Force-no-gen must disable generation even when assets/envs are present.
    monkeypatch.setenv("VOXCLONE_NB_FORCE_NOGEN", "1")
    ns = {}
    exec(guard, ns)
    assert ns["CAN_GENERATE"] is False
    assert "studio" in ns


def test_preflight_wrong_kernel_raises_systemexit(monkeypatch):
    import sys
    from unittest import mock
    guard = next(c.source for c in B.cells
                 if c.cell_type == "code" and "Falscher Kernel" in c.source)
    with mock.patch.dict(sys.modules, {"voxclone.clone": None}):
        with pytest.raises(SystemExit):
            exec(guard, {})
