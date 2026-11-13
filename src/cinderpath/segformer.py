"""Cinderpath SegFormer inference wrapper."""

from typing import List


CLASSES = ["background", "building", "water", "vegetation", "road"]


def argmax(mask: List[float]) -> int:
    return max(range(len(mask)), key=lambda i: mask[i])


def colorize(classes: List[int]) -> List[str]:
    palette = {
        0: "#000000",
        1: "#e4572e",
        2: "#17bebb",
        3: "#76b041",
        4: "#ffc914",
    }
    return [palette.get(cls, "#ffffff") for cls in classes]


def decode_logits(logits: List[List[float]]) -> List[int]:
    return [argmax(row) for row in logits]


def class_names(indices: List[int]) -> List[str]:
    return [CLASSES[i] if 0 <= i < len(CLASSES) else "unknown" for i in indices]
