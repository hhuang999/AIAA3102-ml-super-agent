import inspect

from superagent import training
from superagent.helper_api import reference
from superagent.prompts import blocks, builders


def test_render_uses_live_signature_and_explains_returns(monkeypatch):
    def replacement(model, optimizer, *, new_option=None):
        pass

    monkeypatch.setattr(training, "restore_torch", replacement)
    text = reference()
    assert "restore_torch" + str(inspect.signature(replacement)) in text
    assert "(completed_epoch: int, sampler_state)" in text
    assert "./work is deleted" in text


def test_patch_sees_same_installed_helper_contract():
    op = builders.OpPrompt("refine", "improve", [], "requirements")
    _, user = builders.patch_prompt(op, "change model", "csv", "x=1", "ok")
    assert reference() in user
    assert (
        "inference.json" in user
        and "remove or migrate ALL old checkpoint readers" in user
    )
    assert reference() in builders.reviewer("classification", "x=1", "off", 60)[1]


def test_declared_task_prompts_agree_with_label_free_inference():
    resources = blocks.Resources(100, 1, 100, 1, 0, "off")
    rules = blocks.implementation_rules(resources, 42, .2, declared_split=True)
    assert "without reading validation labels or computing a score" in rules
    assert "must be `VALIDATION_SCORE:" not in rules
    reviewer = builders.reviewer("classification", "x=1", "off", 60, declared_split=True)[1]
    assert "Predict-only needs files outside inference_inputs, reads validation labels, calculates a score" in reviewer
    assert "The score marker line is missing" not in reviewer
    assert "must be `VALIDATION_SCORE:" in blocks.implementation_rules(resources, 42, .2)
