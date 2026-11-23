import re
from collections.abc import Iterable


def build_snippet(text: str, terms: Iterable[str], window: int = 80) -> str:
    clean_terms = sorted({t for t in terms if t}, key=len, reverse=True)
    if not clean_terms or not text:
        return text[: window * 2].replace("\n", " ")

    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in clean_terms) + r")\b",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return text[: window * 2].replace("\n", " ")

    best_center = matches[0].start()
    best_count = 0
    for m in matches:
        c = m.start()
        count = sum(1 for other in matches if abs(other.start() - c) <= window)
        if count > best_count:
            best_count = count
            best_center = c

    start = max(0, best_center - window)
    end = min(len(text), best_center + window)
    fragment = text[start:end].replace("\n", " ")
    highlighted = pattern.sub(lambda m: f"**{m.group(0)}**", fragment)

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{highlighted}{suffix}"