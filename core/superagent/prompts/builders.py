"""Prompt builders for every LLM role. Each returns a (system, user) pair or prompt parts.

Only wording lives here; control flow decides which builder to call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import policy
from .blocks import (
    FULL_SCRIPT_FORMAT,
    LEAKAGE_RULES,
    ROBUSTNESS_ADVICE,
    ROLE_LINE,
    code_block,
    join_sections,
    section,
)

COMPONENTS_TEXT = """## Pipeline components
Every solution script is conceptually split into three components:
- data_prep: read data from ./data, clean it, engineer features, and split into training, validation and test parts.
- model_def: model architecture (including the choice of a pretrained model), loss function and optimiser.
- train_eval: training loop, validation, metric tracking, keeping the best model, predicting the test set and writing the prediction file."""

COMPONENT_DUTIES = {
    "data_prep": "Read the data, clean it, build features (preprocessing, encoding, transforms, suitable augmentation) and split into training, validation and test parts.",
    "model_def": "Design or choose the model for the task and the features from the previous step, and define the loss function and the optimiser.",
    "train_eval": "Train, validate, track the metric, keep the best model, compute the validation score, predict the test set, write the prediction file and save the model files.",
}


@dataclass
class OpPrompt:
    """Everything an operation contributes to the prompts built from it."""

    kind: str
    system: str
    materials: list[str]
    requirements: str
    planner_focus: str = ""
    patch_hint: str = ""
    patch_extra: str = ""
    patch_system_focus: str = ""
    meta: dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- startup roles

def desc_cleaner(raw: str) -> tuple[str, str]:
    system = "You clean machine learning task descriptions. You output only the cleaned description."
    user = join_sections(
        section("Task description", raw),
        """# Instructions
Remove only content unrelated to the essence of the task:
- restrictions on network access, URLs and links,
- time limits and other run limits,
- mentions of the form of the development environment (notebooks, platforms, kernels),
- leaderboard, ranking and prize information.
Keep verbatim: the task goal, the data description, the definition of the evaluation metric, the required format of the prediction file, and anything else relevant to the task.
Output only the cleaned description, with no explanations or preamble.""",
    )
    return system, user


def direction_judge(task_desc: str) -> tuple[str, str]:
    system = "You decide whether an evaluation metric should be minimised or maximised."
    user = join_sections(
        section("Task description", task_desc),
        """# Instructions
Look only at the part of the description about the evaluation metric. Decide whether the metric should be minimised or maximised. Error-like and loss-like metrics (log loss, RMSE, error rate, ...) are minimised; accuracy-like and score-like metrics (accuracy, F1, AUC, ...) are maximised. Answer in JSON with fields "goal" ("minimize" or "maximize") and "explanation".""",
    )
    return system, user


# --------------------------------------------------------------------------- shared material sections

def base_materials(task_desc: str, data_preview: str) -> list[str]:
    return [section("Task description", task_desc), section("Data preview", data_preview)]


def attempts_section(title: str, summary: str) -> str:
    return section(title, summary)


def current_code_section(code: str) -> str:
    return section("Current solution code", code_block(code))


def output_section(trimmed_output: str, title: str = "Output of the current solution") -> str:
    return section(title, "```\n" + trimmed_output.rstrip() + "\n```")


# --------------------------------------------------------------------------- draft

DRAFT_SYSTEM = (
    ROLE_LINE + " Design a complete modelling pipeline for the task (data, model, training, inference). "
    "Implement a model that genuinely learns from the data; every prediction must come from real inference. "
    "Aim for a competitive solution, not a perfunctory baseline."
)


def draft_requirements(rules: str, env: str, catalog_text: str) -> str:
    return join_sections(
        """# Requirements
1. Before designing, answer three questions for yourself: What are the concrete characteristics and difficulties of THIS task (not generic statements such as "this is a classification task")? Which properties of the chosen method address exactly these characteristics? Which result would show the idea is right, and which result would mean the idea must change? The overview should naturally reflect this thinking.
2. Keep the first version relatively simple: no complex model ensembles and no large hyper-parameter searches at this stage.
3. NOVELTY (critical). The new solution must differ substantially from every existing attempt listed above.
   - First analyse the core idea of each existing attempt.
   - Prefer a completely different route that has not been tried yet.
   - You may build on an existing idea only if you add a new idea that fundamentally changes the solution.
   - Small changes are forbidden: different hyper-parameters, a similar model, slightly different preprocessing.
4. The overview must be concise but complete: 8 to 12 sentences saying what, why and how.
5. Use the evaluation metric required by the task description.
6. Do not perform exploratory data analysis.
7. The data is already prepared in ./data; there is nothing to unzip.""",
        catalog_text,
        rules,
        LEAKAGE_RULES,
        env,
        ROBUSTNESS_ADVICE,
    )


def draft_full(task_desc: str, data_preview: str, root_summary: str, requirements: str) -> tuple[str, str]:
    user = join_sections(
        *base_materials(task_desc, data_preview),
        attempts_section("Existing drafts (attempts summary)", root_summary),
        requirements,
        FULL_SCRIPT_FORMAT,
    )
    return DRAFT_SYSTEM, user


STAGE_SPECIFIC = {
    "data_prep": "Responsibility: read the data, clean it, build features (preprocessing, encoding, transforms, suitable data augmentation) and split into training, validation and test parts. It must include both data reading and real feature engineering, not just raw data loading. Do not build a model and do not write training or evaluation code.",
    "model_def": "Responsibility: from the task and the features produced by the previous step, design or choose the model, and define the loss function and the optimiser. Define only these three objects (model, loss, optimiser); no training loop and no data processing code. The model's output must be compatible with the evaluation metric. A self-designed architecture must include regularisation components.",
    "train_eval": "Responsibility: using the data, features, model, loss and optimiser already defined by the previous steps, write the training loop with validation, metric tracking and saving of the best model; then load the best model, compute the validation score, predict the test set, write the prediction file and save the model files. Use the variables and objects defined by previous steps exactly as they are: do not redefine them, reload them or replace them with another approach. The validation score must use the same prediction pipeline as test inference. Actively prevent overfitting (augmentation, early stopping, weight decay, dropout, ...). Implement the task's metric completely, including every threshold, component and aggregation step it requires; no approximate or simplified version. The last printed line is the score marker.",
}


def draft_stage(
    task_desc: str, data_preview: str, root_summary: str, requirements: str, stage: str, previous: list[tuple[str, str, str]], mode: str
) -> tuple[str, str]:
    system = DRAFT_SYSTEM + f" You are now responsible only for the {stage} step: write only this step's code, not the whole solution."
    if previous:
        prev = "\n\n".join(f"## Step: {name}\nPlan: {plan}\n{code_block(code)}" for name, plan, code in previous)
    else:
        prev = "This is the first step."
    specific = STAGE_SPECIFIC[stage]
    if stage == "data_prep" and mode in ("catalog", "open"):
        specific += " " + policy.DATA_PREP_PRETRAINED_HINT
    if stage == "model_def":
        specific += " " + (policy.MODEL_DEF_OFF_HINT if mode == "off" else policy.MODEL_DEF_PRETRAINED_HINT)
    user = join_sections(
        *base_materials(task_desc, data_preview),
        attempts_section("Existing drafts (attempts summary)", root_summary),
        section("Previous steps", prev),
        section("Current step", f"{stage}: {COMPONENT_DUTIES[stage]}"),
        requirements,
        f"""# Requirements for this step
{specific}
Staged-writing rules:
- This step's code must concatenate cleanly with the code of the other steps.
- Keep variable names consistent with the previous steps.
- Do not repeat code that previous steps already contain; assume it exists.
- Handle edge cases carefully.""",
        """# Output format
First 2 to 3 sentences describing what this step does, then exactly one ```python code block containing only this step's code.""",
    )
    return system, user


def draft_integrator(
    task_desc: str, data_preview: str, root_summary: str, requirements: str, steps: list[tuple[str, str, str]]
) -> tuple[str, str]:
    system = ("You are good at writing clean, efficient machine learning code. You have received three pieces of code, "
              "each responsible for one part of a pipeline. Merge them into one coherent script that runs directly.")
    steps_text = "\n\n".join(f"## Step: {name}\nPlan: {plan}\n{code_block(code)}" for name, plan, code in steps)
    user = join_sections(
        *base_materials(task_desc, data_preview),
        attempts_section("Existing drafts (attempts summary)", root_summary),
        section("Steps to merge", steps_text),
        requirements,
        """# Merging rules
- Merge into a single runnable script.
- You are the integrator, not the designer: integrate the steps faithfully and do not introduce models, algorithms or practices that are not in the steps.
- Unify variable names across steps. Remove duplicate imports and definitions.
- When steps conflict, the earlier step's design wins.
- Execution order: data processing, model definition, training and evaluation.
- The final code must print the score marker, write the prediction file and save the model files.
- Assume none of the steps has been executed before: do not skip any step and do not read intermediate artefacts.""",
        """# Output format
First 2 to 3 sentences on how you merged the steps, then exactly one ```python code block containing the complete script.""",
    )
    return system, user


# --------------------------------------------------------------------------- refine / retrospect / borrow

REFINE_SYSTEM = (
    ROLE_LINE + " Make a meaningful improvement to an existing solution. Acceptable: a better model architecture, "
    "feature engineering, training strategy, hyper-parameter optimisation, pipeline improvements. Not acceptable: "
    "cosmetic changes, unjustified small tweaks, or breaking existing functionality."
)

REGULAR_PLAN_STRUCTURE = """The plan must follow this structure:
- Change list. For each change: its category (data augmentation, model architecture, loss function, optimisation, regularisation or training strategy); exactly what changes; why this task needs it.
- Limitation of the current solution: the root cause, and the evidence supporting this diagnosis.
- How the changes solve the problem: mechanism, expected improvement, how several changes work together.
- What stays unchanged: explicitly list the random seed, the data split (identical to the parent solution) and other key parts.
About the number of changes: preferably make a single change so cause and effect stay clear; a few closely related changes may go together; merging improvements that the attempts summary already validated separately is also fine."""


def escalated_block(branch_best: float | None, ok_since: int, all_since: int) -> str:
    best = "n/a" if branch_best is None else f"{branch_best:.4f}"
    return f"""The current line of work has STALLED. Best score in this branch: {best}. Since that best result there have been {ok_since} successful nodes and {all_since} nodes in total without improvement.
Possible changes fall into three levels:
- Level 1 (tuning): model and data unchanged, only the training is adjusted. Examples: hyper-parameters, learning-rate schedule, random seed, post-processing thresholds.
- Level 2 (component): replace one component of the pipeline while keeping the overall paradigm. Examples: new feature engineering, different data augmentation, a different backbone, a different loss, added regularisation.
- Level 3 (paradigm): fundamentally change the approach. Examples: switch between tree models and neural networks, single model to ensemble, redefine the learning objective, pseudo-labelling, self-supervised pre-training.
You MUST propose a level 2 or level 3 change. Level 1 changes are not allowed.
State which level you chose and why, which components will change, and why this can break the stall."""


def refine_op(
    task_desc: str, data_preview: str, summary: str, code: str, trimmed_output: str, rules: str,
    escalated: bool, branch_best: float | None, ok_since: int, all_since: int,
) -> OpPrompt:
    if escalated:
        req_core = escalated_block(branch_best, ok_since, all_since)
    else:
        req_core = REGULAR_PLAN_STRUCTURE
    requirements = join_sections(
        "# Requirements\n" + req_core + """
Common requirements:
- Propose one concrete, actionable improvement.
- It must clearly differ from the attempts already listed in the attempts summary.
- Consider the depth of the improvement: adding or modifying components on the current model (often more effective), or replacing the whole model (only when the current structure is fundamentally unsuitable).
- Keep the plan concise but complete: 8 to 12 sentences. Do not perform exploratory data analysis.""",
        rules,
        LEAKAGE_RULES,
        ROBUSTNESS_ADVICE,
    )
    materials = base_materials(task_desc, data_preview) + [
        attempts_section("Improvements already tried on this solution (attempts summary)", summary),
        current_code_section(code),
        output_section(trimmed_output),
    ]
    return OpPrompt(
        kind="refine",
        system=REFINE_SYSTEM,
        materials=materials,
        requirements=requirements,
        planner_focus="First analyse the task description and the data type carefully; the improvement must target this specific task.",
        patch_system_focus="Focus on a targeted, well-justified improvement.",
    )


def retrospect_op(
    task_desc: str, data_preview: str, summary: str, code: str, trimmed_output: str, rules: str, traj: str,
    escalated: bool, branch_best: float | None, ok_since: int, all_since: int,
) -> OpPrompt:
    if escalated:
        core = escalated_block(branch_best, ok_since, all_since) + """
Analyse the trajectory and find out why progress stopped. State which level you chose, which phenomenon in the trajectory led to this choice, and which concrete change can break the stall."""
    else:
        core = REGULAR_PLAN_STRUCTURE + """
Every item must be grounded in the trajectory:
- The "why" of each change must cite the trajectory.
- The limitation analysis must point to the pattern in the trajectory that reveals the problem, citing concrete steps and scores as evidence.
- The solution idea must say which success or failure in the trajectory it learns from.
- "What stays unchanged" must list practices the trajectory proved effective and that should be kept."""
    requirements = join_sections(
        "# Requirements\n" + core + """
Learning from the trajectory: identify what worked and what did not; build on successes; avoid directions that failed repeatedly; understand why certain changes worked and exploit that. Principle: learn from history instead of changing things at random.
Common requirements: one concrete improvement; it must differ from the attempts in the attempts summary; rely mainly on the trajectory; no exploratory data analysis.""",
        ROBUSTNESS_ADVICE,
        rules,
    )
    materials = base_materials(task_desc, data_preview) + [
        attempts_section("Improvements already tried on this solution (attempts summary)", summary),
        current_code_section(code),
        output_section(trimmed_output),
        section("Trajectory of this branch (oldest first)", traj),
    ]
    return OpPrompt(
        kind="retrospect",
        system=REFINE_SYSTEM + " Use the history of this branch to decide what to change.",
        materials=materials,
        requirements=requirements,
        planner_focus="Analyse the trajectory carefully: what was tried, what worked and what did not. Continue successful practices and avoid repeating failed ones.",
        patch_hint="Modify the code according to the trajectory: continue the practices that succeeded and avoid those that failed.",
        patch_extra=section("Trajectory of this branch (oldest first)", traj),
        patch_system_focus="Ground the change in the branch's history.",
    )


def borrow_op(
    task_desc: str, data_preview: str, plan: str, code: str, trimmed_output: str, rules: str, references: str,
) -> OpPrompt:
    requirements = join_sections(
        """# Requirements
Task: analyse the reference solutions, learn from their strengths, and improve the CURRENT solution. Do not copy a reference solution.
Plan structure:
- Change list. For each change: the specific technique taken from a reference; why this task needs it; which reference it comes from.
- Limitation of the current solution and how the reference solves it: root cause, comparative evidence.
- How the reference technique applies to the current solution: mechanism, expected improvement.
- What stays unchanged.
Bring in exactly ONE technique, the most promising one. More references help you choose better; they are not an invitation to transplant more things. Do not stitch the references' practices together.
Focus on understanding why the reference succeeded and whether its practice addresses the current solution's specific limitation.""",
        rules,
    )
    materials = base_materials(task_desc, data_preview) + [
        section("Current solution design", plan),
        current_code_section(code),
        output_section(trimmed_output),
        section("Reference solutions from other branches", references),
    ]
    return OpPrompt(
        kind="borrow",
        system=REFINE_SYSTEM + " Improve the current solution by borrowing one technique from stronger solutions in other branches.",
        materials=materials,
        requirements=requirements,
        planner_focus="Analyse carefully why the reference solutions succeeded. With several references, compare them first, then choose the single most relevant one.",
        patch_hint="Bring in exactly one technique from the reference solution(s) as the plan says; do not copy the reference wholesale.",
        patch_extra=section("Reference solutions from other branches", references),
        patch_system_focus="Transfer one technique from a reference solution.",
    )


# --------------------------------------------------------------------------- repair

REPAIR_SYSTEM = (
    ROLE_LINE + " The previous solution either crashed, did not produce the required output files, or produced output "
    "in the wrong format. Debug it systematically: read the error, find the root cause, and make the smallest targeted "
    "fix. Keep the original design intent and code quality. Do not change things at random, do not delete large blocks "
    "of code, do not replace the model with fake predictions, and do not take shortcuts. The code will run in a fresh "
    "environment and must be self-contained."
)


def repair_op(task_desc: str, data_preview: str, code: str, trimmed_output: str, rules: str, experience: str) -> OpPrompt:
    requirements = join_sections(
        """# Requirements
- First explain in two or three sentences how you will fix the problem.
- Do not perform exploratory data analysis.
- The common libraries are stable and installed; the problem is not caused by library versions. Do not suggest reinstalling anything.""",
        rules,
        ROBUSTNESS_ADVICE,
    )
    materials = base_materials(task_desc, data_preview) + [
        current_code_section(code),
        output_section(trimmed_output, "Output of the failed run (including the checker's notes)"),
    ]
    if experience:
        materials.append(section("Similar errors fixed earlier (for reference)", experience))
    return OpPrompt(kind="repair", system=REPAIR_SYSTEM, materials=materials, requirements=requirements,
                    patch_system_focus="Fix the error with minimal, targeted changes.")


def repair_from_plan(task_desc: str, data_preview: str, plan: str, rules: str) -> tuple[str, str]:
    user = join_sections(
        *base_materials(task_desc, data_preview),
        section("Plan of the solution to implement", plan or "(no plan was recorded; design a sound solution for the task)"),
        "# Requirements\nThe previous attempt produced no usable code. Write a complete, self-contained script that implements the plan above.",
        rules,
        ROBUSTNESS_ADVICE,
        FULL_SCRIPT_FORMAT,
    )
    return REPAIR_SYSTEM, user


# --------------------------------------------------------------------------- synthesis

SYNTHESIS_SYSTEM = (
    ROLE_LINE + " You are given the best solutions of several independent lines of work. Your task is to synthesise "
    "these different approaches into a brand-new solution. This is a fresh start whose goal is to create new ideas by "
    "combining insights from different successful directions."
)


def synthesis_prompt(task_desc: str, data_preview: str, reps: str, rules: str) -> tuple[str, str]:
    user = join_sections(
        *base_materials(task_desc, data_preview),
        section("Best solution of each branch", reps),
        """# Requirements
- Analyse the key techniques and practices behind each representative's success.
- You are not improving one existing solution; you are creating a new one.
- Look for synergies between the successful techniques of different branches.
- The new solution must be original, not a simple concatenation.
- Give an overview of the solution, then a single runnable, complete script.
- Do not perform exploratory data analysis.""",
        rules,
        FULL_SCRIPT_FORMAT,
    )
    return SYNTHESIS_SYSTEM, user


# --------------------------------------------------------------------------- full-script versions of operations

def op_full_script(op: OpPrompt, plan_text: str | None = None, emphasise_complete: bool = False) -> tuple[str, str]:
    extra = ""
    if plan_text:
        extra = section("Improvement plan", plan_text + "\n\nModify the current solution according to this plan and output the complete script.")
    note = ""
    if emphasise_complete:
        note = "# Important\nOutput the COMPLETE, self-contained script. Do not omit any part and do not use placeholders such as '... rest unchanged ...'."
    user = join_sections(*op.materials, extra, op.requirements, note, FULL_SCRIPT_FORMAT)
    return op.system, user


# --------------------------------------------------------------------------- planner

PLANNER_TASK = """# Task
Based on the task requirements, the data characteristics and the run results, analyse the current solution, find the improvement opportunities most likely to raise the final test performance, then choose the components to modify and give a detailed, actionable modification plan."""

PLANNER_CRITERIA = """# Selection criteria
- Prefer only 1 to 2 components; at most 3.
- Choose the components most likely to bring an improvement.
- Never change the evaluation metric or the prediction file format.
- Never break interfaces or variable names shared between components.
- Use the run output, performance bottlenecks and stability problems to guide the choice.
- Focus on improvements that generalise to the test set, not only on raising the validation score.
- Write each component's plan as: what to change, why, how, interface constraints."""

PLANNER_FIELDS = """# Output fields (JSON)
- "rationale": string, 1 to 4 sentences: why these components are the most worthwhile to change now, including the root-cause analysis.
- "components": array of 1 to 3 strings, each one of "data_prep", "model_def", "train_eval".
- "changes": object whose keys are exactly the names in "components"; each value is a 2 to 5 sentence plan stating what changes (the specific technique and its category), why (the root cause and why this task needs it), how to implement it, and the interface constraints (which variable names and function signatures must stay unchanged).
Return only the JSON object."""


def planner_single(op: OpPrompt) -> tuple[str, str]:
    user = join_sections(*op.materials, op.requirements, COMPONENTS_TEXT,
                         PLANNER_TASK + ("\n" + op.planner_focus if op.planner_focus else ""), PLANNER_CRITERIA, PLANNER_FIELDS)
    return op.system, user


def planner_free(op: OpPrompt) -> tuple[str, str]:
    user = join_sections(
        *op.materials,
        op.requirements,
        COMPONENTS_TEXT,
        """# Output
Identify at most 4 key points to modify. Write one paragraph explaining which components or aspects you intend to change, why they need to change (based on the run results and the task requirements), and the strategy. Name which of the pipeline components above are involved. Output plain text only: write only the plan, with no complete code, no code blocks and no run logs.""",
    )
    return op.system, user


def planner_second(
    task_desc: str, data_preview: str, code: str, trimmed_output: str, free_plan: str, experience: str, have_records: bool, system: str
) -> tuple[str, str]:
    if have_records:
        guide = """# Task
- Turn the preliminary plan into structured, component-specific modifications.
- Use the past experience to refine the plan: borrow practices that worked, avoid practices that failed.
- DO NOT deviate from the core direction of the preliminary plan. Past experience refines and strengthens it; it does not replace it.
- "rationale" must say how the past experience influenced this refinement."""
    else:
        guide = """# Task
Convert the preliminary plan directly into structured, component-specific modifications, strictly following its direction."""
    user = join_sections(
        *base_materials(task_desc, data_preview),
        current_code_section(code),
        output_section(trimmed_output),
        section("Preliminary plan", free_plan),
        section("Past experience", experience),
        COMPONENTS_TEXT,
        guide,
        PLANNER_CRITERIA,
        PLANNER_FIELDS,
    )
    return system, user


# --------------------------------------------------------------------------- patch coder

PATCH_FORMAT = """# Patch format
<<<<<<< SEARCH
original code to be replaced, copied verbatim
=======
new code
>>>>>>> REPLACE"""


def patch_prompt(
    op: OpPrompt, plan_text: str, data_preview: str, code: str, trimmed_output: str, retry_note: str = ""
) -> tuple[str, str]:
    system = "Modify an existing solution according to the given improvement plan so that its test performance improves. " + op.patch_system_focus
    hint = f"\n- {op.patch_hint}" if op.patch_hint else ""
    reqs = f"""# Requirements
- You must implement the improvement plan. Modify only the components and aspects the plan names.{hint}
- Keep the interfaces, variable names and compatibility constraints mentioned in the plan.
- Make minimal, targeted modifications.
- Each SEARCH part must be copied verbatim from the code above: indentation, whitespace, line breaks and comments must match exactly. Do not retype from memory and do not change indentation.
- Keep each SEARCH part short: only the lines to be replaced, plus minimal context if needed. Prefer several small blocks over one large block.
- Every SEARCH part needs a REPLACE part. Do not put other text between blocks.
- The modified file must still run.
- Never output the whole file. A reply without patch blocks is rejected."""
    user = join_sections(
        retry_note,
        section("Improvement plan", plan_text),
        section("Data preview", data_preview),
        current_code_section(code),
        output_section(trimmed_output),
        op.patch_extra,
        reqs,
        PATCH_FORMAT,
        "# Output format\nYou may start with a few sentences of explanation, then the patch blocks.",
    )
    return system, user


def repair_patch_prompt(op: OpPrompt, data_preview: str, code: str, trimmed_output: str, retry_note: str = "") -> tuple[str, str]:
    reqs = """# Requirements for the patch
- Make minimal, targeted modifications that fix the error.
- Each SEARCH part must be copied verbatim from the code above: indentation, whitespace, line breaks and comments must match exactly.
- Keep each SEARCH part short; prefer several small blocks over one large block.
- Every SEARCH part needs a REPLACE part. Do not put other text between blocks.
- The modified file must still run.
- Never output the whole file. A reply without patch blocks is rejected."""
    user = join_sections(retry_note, *op.materials, op.requirements, reqs, PATCH_FORMAT,
                         "# Output format\nFirst two or three sentences explaining the fix, then the patch blocks.")
    return op.system, user


# --------------------------------------------------------------------------- review / assessment roles

def reviewer(task_desc: str, code: str, mode: str, run_limit_secs: float) -> tuple[str, str]:
    system = "You are a senior data-science code reviewer. You only look for data leakage and critical correctness problems."
    trust = policy.REVIEW_OFF if mode == "off" else policy.REVIEW_TRUST_CATALOG
    user = join_sections(
        section("Task description", task_desc),
        section("Code to review", code_block(code)),
        f"""# Environment facts you must trust without question
- ./data, ./work and ./out all exist. Do not question paths.
- The prediction file location is ./out/predictions.csv.
- All libraries are installed and recent enough.
- The script's time limit is {int(run_limit_secs)} seconds.
- {trust if mode != "off" else "There is no network access."}""",
        (f"# Strictly forbidden\n- {trust}" if mode != "off" else f"# Pretrained policy\n- {trust}"),
        """# Highest priority: data leakage
- Processing order: scaling, dimensionality reduction, target encoding or similar transforms fitted BEFORE the split, or over-sampling before the split. This must be fixed. Correct order: split first, fit on the training part only, then transform each part.
- Split strategy: raise it only when the split method mathematically violates the dependency structure of the data (for example samples of the same group on both sides, or time-ordered data shuffled randomly).

# Second priority: critical correctness
- The metric used differs from the one the task requires.
- Predictions do not come from model inference (all zeros, all ones, training mean, random numbers).
- The test set is predicted with the last-epoch model instead of the model with the best validation performance.
- API pitfalls: LightGBM early stopping must use callbacks; XGBoost early_stopping_rounds belongs to the constructor; AdamW comes from torch.optim.
- Progress bars or per-batch logging are used.
- The score marker line is missing; the prediction file path is wrong; model files are not saved.

# Decision rule
Ask for a revision ONLY if you found one of the concrete problems above. If you found no concrete logic problem, pass. The default is pass.

# Output fields (JSON)
- "verdict": "pass" or "revise".
- "findings": 2 to 4 sentences: what you found, why it matters and how it will be fixed; when there is no problem, one sentence with the conclusion.
- "patch": when verdict is "revise", patch blocks in this exact format containing only the lines to fix; when verdict is "pass", null.
<<<<<<< SEARCH
original lines copied verbatim
=======
fixed lines
>>>>>>> REPLACE""",
    )
    return system, user


def result_reader(code: str, trimmed_output: str) -> tuple[str, str]:
    system = "You wrote this code to solve the task and now evaluate its run output."
    user = join_sections(
        section("Code", code_block(code)),
        output_section(trimmed_output, "Run output"),
        """# Instructions
Decide whether the run failed, and report what was actually observed. Focus only on real runtime errors, exceptions or crashes (warnings are not errors).

# Output fields (JSON)
- "has_error": boolean, true if the output shows the run failed or hit an error.
- "observations": 2 to 3 sentences. On success: the key experimental results. On failure: the error encountered. Observations only, no suggestions.
- "score": number or null: the validation score when the run succeeded.
- "goal": "minimize" or "maximize": whether the reported score is better when lower or when higher.
- "method_summary": summary of what the code does in each stage: data preprocessing, feature engineering, model architecture and training, validation method.""",
    )
    return system, user


def leak_checker(task_desc: str, code: str, trimmed_output: str, score: float, higher_is_better: bool) -> tuple[str, str]:
    system = ("You are a machine learning engineer specialised in detecting data leakage. Decide whether this code has a "
              "leak that inflates the validation score while the test performance would be poor.")
    user = join_sections(
        section("Task description", task_desc),
        section("Code", code_block(code)),
        output_section(trimmed_output, "Run output"),
        f"""# Score
Validation score: {score}. The metric is {"higher" if higher_is_better else "lower"} is better.

# Common leakage forms
- Validation or test data used during training (for example a transformer fitted on all data).
- An improper split (temporal leakage in time series, group leakage).
- Feature engineering using global statistics that include validation or test data.
- Data augmentation copying validation samples into the training set.
- Target information or future information unavailable at prediction time.
Note: some tasks really are easy and a near-perfect score can be legitimate. Consider the task difficulty before concluding.

# Output fields (JSON)
- "leak_found": boolean.
- "evidence": if a leak exists, the exact code construct and location causing it; otherwise why this high score is plausible.
- "certainty": "high" (clear evidence in the code), "medium" (a suspicious construct likely to leak) or "low" (the task may simply be easy, or the code is unclear).""",
    )
    return system, user


def format_fixer(sample_head: str, pred_head: str, sample_cols: list[str], pred_cols: list[str]) -> tuple[str, str]:
    system = "You fix the column names of a prediction file so they match the required sample file."
    user = join_sections(
        section("Sample file (first rows)", "```\n" + sample_head + "\n```"),
        section("Current prediction file (first rows)", "```\n" + pred_head + "\n```"),
        f"""# Column names
Sample columns: {sample_cols}
Current columns: {pred_cols}

# Instructions
Assign a sample column name to each column of the current file. Use every sample column name exactly once. Names differing only in case, spaces or underscores, or with the same meaning, map to each other; for names that differ a lot, infer from the type, range and pattern of the values. Only rename columns: do not change the data and do not reorder columns.

# Output fields (JSON)
- "columns": array of strings; element i is the sample column name for column i of the current file.""",
    )
    return system, user
