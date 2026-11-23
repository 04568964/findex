import re
from dataclasses import dataclass

from findex.models import Index

_TOKEN_RE = re.compile(r'\s*(?:(\()|(\))|"([^"]+)"|([^\s()]+))')


class QueryNode:
    def __and__(self, other: "QueryNode") -> "And":
        return And(self, other)

    def __or__(self, other: "QueryNode") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)

    def evaluate(self, index: Index) -> set[int]:
        raise NotImplementedError

    def extract_terms(self) -> list[str]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Term(QueryNode):
    value: str

    def evaluate(self, index: Index) -> set[int]:
        if self.value not in index:
            return set()
        return {p.doc_id for p in index[self.value]}

    def extract_terms(self) -> list[str]:
        return [self.value]


@dataclass(frozen=True, slots=True)
class Phrase(QueryNode):
    terms: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        if not self.terms:
            return set()
        if not index.has_positions:
            raise RuntimeError("Index was built without positions; phrase queries are unavailable")
        if any(t not in index for t in self.terms):
            return set()

        posting_maps = [{p.doc_id: set(p.positions) for p in index[t]} for t in self.terms]
        common_docs: set[int] = set(posting_maps[0].keys())
        for m in posting_maps[1:]:
            common_docs.intersection_update(m.keys())

        matched: set[int] = set()
        for doc_id in common_docs:
            start_positions = posting_maps[0][doc_id]
            for pos in start_positions:
                if all((pos + offset) in posting_maps[offset][doc_id] for offset in range(1, len(self.terms))):
                    matched.add(doc_id)
                    break
        return matched

    def extract_terms(self) -> list[str]:
        return list(self.terms)


@dataclass(frozen=True, slots=True)
class And(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def extract_terms(self) -> list[str]:
        return self.left.extract_terms() + self.right.extract_terms()


@dataclass(frozen=True, slots=True)
class Or(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def extract_terms(self) -> list[str]:
        return self.left.extract_terms() + self.right.extract_terms()


@dataclass(frozen=True, slots=True)
class Not(QueryNode):
    operand: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return set(index.docs.keys()) - self.operand.evaluate(index)

    def extract_terms(self) -> list[str]:
        return []


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self) -> tuple[str, str]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> QueryNode:
        if not self.tokens:
            raise ValueError("Empty query")
        node = self._parse_or()
        if self.peek() is not None:
            raise ValueError(f"Unexpected token: {self.peek()}")
        return node

    def _parse_or(self) -> QueryNode:
        node = self._parse_and()
        while (tok := self.peek()) is not None and tok == ("OP", "OR"):
            self.consume()
            node = Or(node, self._parse_and())
        return node

    def _parse_and(self) -> QueryNode:
        node = self._parse_not()
        while (tok := self.peek()) is not None:
            if tok == ("OP", "AND"):
                self.consume()
                node = And(node, self._parse_not())
            elif tok[0] in ("TERM", "PHRASE", "LPAREN") or tok == ("OP", "NOT"):
                node = And(node, self._parse_not())
            else:
                break
        return node

    def _parse_not(self) -> QueryNode:
        if self.peek() == ("OP", "NOT"):
            self.consume()
            return Not(self._parse_not())
        return self._parse_primary()

    def _parse_primary(self) -> QueryNode:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of query")
        kind, value = tok
        if kind == "LPAREN":
            self.consume()
            node = self._parse_or()
            if self.peek() != ("RPAREN", ")"):
                raise ValueError("Missing closing parenthesis")
            self.consume()
            return node
        if kind == "PHRASE":
            self.consume()
            parts = tuple(w.lower() for w in value.split() if w)
            return Phrase(parts)
        if kind == "TERM":
            self.consume()
            return Term(value.lower())
        raise ValueError(f"Unexpected token: {tok}")


def _lex(query: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(query):
        m = _TOKEN_RE.match(query, pos)
        if not m:
            raise ValueError(f"Invalid syntax near: {query[pos:]!r}")
        pos = m.end()
        lparen, rparen, phrase, word = m.groups()
        if lparen:
            tokens.append(("LPAREN", "("))
        elif rparen:
            tokens.append(("RPAREN", ")"))
        elif phrase is not None:
            tokens.append(("PHRASE", phrase))
        elif word:
            if word in ("AND", "OR", "NOT"):
                tokens.append(("OP", word))
            else:
                tokens.append(("TERM", word))
    return tokens


def parse_query(query: str) -> QueryNode:
    return _Parser(_lex(query)).parse()