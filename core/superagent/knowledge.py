"""Task-owned, sourced method cards selected by observed modality."""

from pathlib import Path

import yaml

from .errors import InvalidInputError


def load_cards(task: Path, modalities):
    path = task / "knowledge.yaml"
    if not path.exists():
        return []
    try:
        cards = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(cards, list):
            raise ValueError("expected a list of method cards")
        selected = []
        for card in cards:
            fields = ("id", "version", "modality", "source", "method", "counterexample")
            if not isinstance(card, dict) or any(
                not isinstance(card.get(k), str) or not card[k].strip() for k in fields
            ):
                raise ValueError(
                    f"each card needs nonempty strings: {', '.join(fields)}"
                )
            if card["modality"] in (*modalities, "classification"):
                selected.append(card)
        ids = [(card["id"], card["version"]) for card in selected]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate method card ID and version")
        return selected
    except (ValueError, yaml.YAMLError) as exc:
        raise InvalidInputError(f"{path}: {exc}") from exc


def read_cards(task: Path, modalities):
    return "\n\n".join(
        "{id}@{version}: {method}\nAvoid when: {counterexample}\nSource: {source}".format(**card)
        for card in load_cards(task, modalities)[:6]
    )
