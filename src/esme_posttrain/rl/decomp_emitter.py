"""Emit grpo-decomp ``CompletionSet`` artifacts from an Esme checkpoint.

The decomposition harness (``grpo-decomp``) grades a run by comparing three arms —
``base``, ``correct``, ``random`` — on one held-out set. Each arm is a ``CompletionSet``:
a ``provenance.json`` + ``completions.jsonl`` pair, one line per problem, each line a
``{problem: {id, question, gold_answer}, samples: [...]}`` record with a uniform number
of samples per problem.

This module builds those artifacts from an Esme Countdown-Lite held-out set and a local
Esme bundle, reusing the same generation path the acceptance eval uses (``model.generate``
+ ``extract_candidate_expression``). The repos exchange artifacts, never code, so the
schema is reproduced here rather than imported from grpo-decomp; the two sides agree on
the contract by construction (and a fixture test pins it).

Contract with the grpo-decomp ``esme-countdown`` task-set (see grpo-decomp
``llm_grpo_gains/esme_countdown.py``):

- ``problem.id``          = the held-out task id (``countdown_heldout_fresh_easy_0000``).
- ``problem.question``    = the held-out task prompt (verbatim).
- ``problem.gold_answer`` = ``target=<t>;numbers=<n1,n2,...>`` (numbers sorted ascending).
- ``dataset``             = ``{name: esme-countdown, config: <set>, split: <set>,
                              revision: <content hash of the problems>}``.

Each written ``sample`` is the model's extracted Countdown expression wrapped in
``\\boxed{...}`` so the harness's strict/lenient extractor recovers exactly that
expression before the registered Esme verifier grades it with Esme's own rules
(each supplied number used exactly once, only ``+ - *``, evaluates to the target).
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from esme_posttrain.bundle import load_dense_backbone_bundle
from esme_posttrain.rl.countdown_lite import (
    extract_candidate_expression,
    render_chat_prompt,
)

PROVENANCE_FILE = "provenance.json"
COMPLETIONS_FILE = "completions.jsonl"

#: The ``DatasetRef.name`` both the emitter and the grpo-decomp verifier key on.
ESME_COUNTDOWN_SOURCE = "esme-countdown"

#: Packages whose versions are stamped into provenance for reproducibility.
_PROVENANCE_PACKAGES = ("torch", "tokenizers")


class DecompEmitterError(ValueError):
    pass


@dataclass(frozen=True)
class EmitRequest:
    bundle_path: Path
    heldout_manifest_path: Path
    output_dir: Path
    set_name: str = "heldout_fresh"
    n: int = 1
    temperature: float = 0.0
    max_new_tokens: int = 12
    seed: int = 0
    device: str = "cpu"
    model_label: str | None = None


def format_countdown_key(numbers: list[int] | tuple[int, ...], target: int) -> str:
    """Encode the answer key as ``target=<t>;numbers=<n1,n2,...>`` (numbers sorted).

    Mirrors grpo-decomp's ``format_countdown_key`` so one parser on the grpo-decomp side
    decodes what this emitter writes.
    """
    sorted_numbers = ",".join(str(int(n)) for n in sorted(int(x) for x in numbers))
    return f"target={int(target)};numbers={sorted_numbers}"


def emit_completion_set(request: EmitRequest) -> dict[str, Any]:
    """Generate completions for a held-out set and write a grpo-decomp ``CompletionSet``.

    Returns a summary dict with the written paths and shape counts.
    """
    if request.n <= 0:
        raise DecompEmitterError("n must be positive")
    if request.max_new_tokens <= 0:
        raise DecompEmitterError("max_new_tokens must be positive")
    if request.temperature < 0.0:
        raise DecompEmitterError("temperature must be non-negative")

    problems = _load_heldout_problems(request.heldout_manifest_path, request.set_name)
    if not problems:
        raise DecompEmitterError(f"no held-out problems found for set {request.set_name!r}")

    target_device = torch.device(request.device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise DecompEmitterError("emitter requested CUDA, but CUDA is not available")
    loaded = load_dense_backbone_bundle(request.bundle_path, map_location=target_device)
    model = loaded.model
    tokenizer = loaded.tokenizer
    device = next(model.parameters()).device
    eos_token_id = tokenizer.token_to_id("<eos>")

    items: list[dict[str, Any]] = []
    torch.manual_seed(request.seed)
    for problem_index, problem in enumerate(problems):
        prompt = render_chat_prompt(str(problem["question"]))
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        if not prompt_ids:
            raise DecompEmitterError(f"problem {problem['id']} produced an empty prompt")
        # A fresh per-problem seed keeps the emitted set reproducible and independent of
        # the number of problems generated before it.
        torch.manual_seed(request.seed + problem_index * 10_000)
        samples = _generate_samples(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            n=request.n,
            temperature=request.temperature,
            max_new_tokens=request.max_new_tokens,
            eos_token_id=eos_token_id,
            device=device,
        )
        items.append(
            {
                "problem": {
                    "id": problem["id"],
                    "question": problem["question"],
                    "gold_answer": problem["gold_answer"],
                },
                "samples": samples,
            }
        )

    dataset_ref = _dataset_ref(request.set_name, problems)
    provenance = _provenance(request, dataset_ref, model_label=_model_label(request, loaded))
    return _write_completion_set(request.output_dir, provenance, items)


def _generate_samples(
    *,
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    n: int,
    temperature: float,
    max_new_tokens: int,
    eos_token_id: int | None,
    device: torch.device,
) -> list[str]:
    """Sample ``n`` completions for one prompt, boxed for the harness's extractor.

    Reuses the acceptance eval's ``model.generate`` path. Each completion string is the
    model's Countdown expression wrapped in ``\\boxed{...}`` — the answer format the
    grpo-decomp strict/lenient extractor reads before the Esme verifier grades it. An
    output with no recoverable expression is written as an empty box, which grades wrong.
    """
    input_tensor = torch.tensor([prompt_ids] * n, dtype=torch.long, device=device)
    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=eos_token_id,
        )
    samples: list[str] = []
    for generated_row in generated.detach().cpu().tolist():
        completion_ids = _truncate_at_eos(generated_row[len(prompt_ids) :], eos_token_id)
        output = tokenizer.decode(completion_ids, skip_special_tokens=False)
        expression = extract_candidate_expression(output)
        samples.append(f"\\boxed{{{expression if expression is not None else ''}}}")
    return samples


def _truncate_at_eos(token_ids: list[int], eos_token_id: int | None) -> list[int]:
    if eos_token_id is None:
        return token_ids
    try:
        eos_index = token_ids.index(eos_token_id)
    except ValueError:
        return token_ids
    return token_ids[:eos_index]


def _load_heldout_problems(manifest_path: Path, set_name: str) -> list[dict[str, str]]:
    """Load one held-out split into canonical ``{id, question, gold_answer}`` records.

    Reads the split file directly from the held-out manifest so the emitter needs no
    generator import and stays faithful to the committed data. Confines the split path to
    the manifest's data root.
    """
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_root = manifest_path.parent.parent
    problems: list[dict[str, str]] = []
    matched = False
    for data_file in manifest["data_files"]:
        path = (manifest_path.parent / str(data_file["path"])).resolve()
        if not path.is_relative_to(data_root):
            raise DecompEmitterError(f"held-out data file escapes data root: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("split") != set_name:
                    continue
                matched = True
                problems.append(
                    {
                        "id": str(row["task_id"]),
                        "question": str(row["prompt"]),
                        "gold_answer": format_countdown_key(row["numbers"], int(row["target"])),
                    }
                )
    if not matched:
        raise DecompEmitterError(
            f"set {set_name!r} not present in held-out manifest {manifest_path}"
        )
    return problems


def _content_revision(set_name: str, problems: list[dict[str, str]]) -> str:
    """A content hash pinning the exact problem records the arms were generated over."""
    digest = hashlib.sha256()
    for problem in problems:
        digest.update(
            f"{problem['id']}\t{problem['question']}\t{problem['gold_answer']}\n".encode()
        )
    return f"{set_name}+{digest.hexdigest()[:16]}"


def _dataset_ref(set_name: str, problems: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "name": ESME_COUNTDOWN_SOURCE,
        "config": set_name,
        "split": set_name,
        "revision": _content_revision(set_name, problems),
    }


def _model_label(request: EmitRequest, loaded: Any) -> str:
    if request.model_label:
        return request.model_label
    manifest = getattr(loaded.bundle, "manifest", None)
    if isinstance(manifest, dict):
        model = manifest.get("model")
        if isinstance(model, dict) and isinstance(model.get("id"), str):
            return str(model["id"])
    return str(request.bundle_path)


def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for name in _PROVENANCE_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def _provenance(
    request: EmitRequest, dataset_ref: dict[str, Any], *, model_label: str
) -> dict[str, Any]:
    return {
        "model": model_label,
        "model_revision": None,
        "backend": "transformers",
        "prompt_strategy": "esme_countdown_chat",
        "sampling": {
            "temperature": float(request.temperature),
            "top_p": 1.0,
            "max_new_tokens": int(request.max_new_tokens),
            "n": int(request.n),
            "seed": int(request.seed),
        },
        "dataset": dataset_ref,
        "n_problems": None,  # filled by _write_completion_set once items are counted
        "commit": "esme-posttrain",
        "dirty": False,
        "python_version": sys.version.split()[0],
        "package_versions": _package_versions(),
    }


def _write_completion_set(
    output_dir: Path, provenance: dict[str, Any], items: list[dict[str, Any]]
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {**provenance, "n_problems": len(items)}
    (output_dir / PROVENANCE_FILE).write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    lines = [json.dumps(item, sort_keys=True) for item in items]
    (output_dir / COMPLETIONS_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "provenance_path": str(output_dir / PROVENANCE_FILE),
        "completions_path": str(output_dir / COMPLETIONS_FILE),
        "set_name": provenance["dataset"]["config"],
        "dataset_name": provenance["dataset"]["name"],
        "dataset_revision": provenance["dataset"]["revision"],
        "n_problems": len(items),
        "n_samples": provenance["sampling"]["n"],
        "model": provenance["model"],
    }
