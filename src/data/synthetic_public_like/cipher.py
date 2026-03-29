from __future__ import annotations

import json
import random
import string
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Support direct execution from the repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.synthetic_public_like.common import ALICE_HEADER


NOUNS = [
    "alice",
    "bird",
    "cat",
    "dragon",
    "hatter",
    "king",
    "knight",
    "mouse",
    "princess",
    "queen",
    "rabbit",
    "student",
    "teacher",
    "turtle",
    "wizard",
]
VERBS = [
    "chases",
    "creates",
    "discovers",
    "dreams",
    "explores",
    "follows",
    "found",
    "imagines",
    "reads",
    "sees",
    "studies",
    "watches",
    "writes",
]
ADJECTIVES = [
    "ancient",
    "bright",
    "clever",
    "colorful",
    "curious",
    "dark",
    "golden",
    "hidden",
    "magical",
    "mysterious",
    "silver",
    "strange",
    "wise",
]
LOCATIONS = [
    "book",
    "castle",
    "cave",
    "crystal",
    "door",
    "forest",
    "garden",
    "island",
    "key",
    "library",
    "map",
    "message",
    "mountain",
    "ocean",
    "palace",
    "potion",
    "puzzle",
    "secret",
    "story",
    "tower",
    "treasure",
    "valley",
    "village",
    "wonderland",
]
PREPOSITIONS = ["above", "around", "beyond", "inside", "near", "through", "under"]


def _build_substitution(rng: random.Random) -> dict[str, str]:
    alphabet = list(string.ascii_lowercase)
    shuffled = alphabet[:]
    rng.shuffle(shuffled)
    return dict(zip(alphabet, shuffled))


def _encrypt_word(word: str, mapping: dict[str, str]) -> str:
    return "".join(mapping.get(char, char) for char in word)


def _make_plain_sentence(rng: random.Random) -> list[str]:
    """Use a few simple sentence templates that match the public plaintext style."""
    template = rng.choice(
        [
            "noun_verb_object",
            "noun_verb_prep_location",
            "the_adj_noun_verb",
            "noun_verb_the_adj_object",
            "noun_verb_the_object",
        ]
    )
    if template == "noun_verb_object":
        return [rng.choice(NOUNS), rng.choice(VERBS), rng.choice(LOCATIONS + NOUNS)]
    if template == "noun_verb_prep_location":
        return [rng.choice(NOUNS), rng.choice(VERBS), rng.choice(PREPOSITIONS), rng.choice(LOCATIONS)]
    if template == "the_adj_noun_verb":
        return ["the", rng.choice(ADJECTIVES), rng.choice(NOUNS), rng.choice(VERBS)]
    if template == "noun_verb_the_adj_object":
        return [rng.choice(NOUNS), rng.choice(VERBS), "the", rng.choice(ADJECTIVES), rng.choice(LOCATIONS)]
    return [rng.choice(NOUNS), rng.choice(VERBS), "the", rng.choice(LOCATIONS)]


def make_cipher_puzzle(rng: random.Random, example_index: int) -> dict[str, Any]:
    """Generate a per-puzzle monoalphabetic substitution cipher task."""
    substitution = _build_substitution(rng)
    example_count = rng.randint(3, 5)

    examples: list[tuple[str, str]] = []
    for _ in range(example_count):
        plain_sentence = " ".join(_make_plain_sentence(rng))
        cipher_sentence = " ".join(_encrypt_word(word, substitution) for word in plain_sentence.split())
        examples.append((cipher_sentence, plain_sentence))

    query_plain = " ".join(_make_plain_sentence(rng))
    query_cipher = " ".join(_encrypt_word(word, substitution) for word in query_plain.split())

    example_lines = "\n".join(f"{cipher} -> {plain}" for cipher, plain in examples)
    prompt = (
        f"{ALICE_HEADER}secret encryption rules are used on text. Here are some examples:\n"
        f"{example_lines}\n"
        f"Now, decrypt the following text: {query_cipher}"
    )
    return {
        "id": f"syn_public_cipher_{example_index:06d}",
        "prompt": prompt,
        "answer": query_plain,
        "category": "cipher",
        "source": "synthetic_public_like_v1",
        "generator_family": "monoalphabetic_substitution",
    }


if __name__ == "__main__":
    print(json.dumps(make_cipher_puzzle(random.Random(42), 0), ensure_ascii=False, indent=2))
