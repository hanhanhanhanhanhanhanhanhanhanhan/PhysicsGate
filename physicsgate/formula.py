"""Chemical formula parsing and sparse composition encodings.

The ESTM paper defines a 100-dimensional sparse composition encoding from H to
Fm. This module intentionally implements only formula parsing and sparse
fraction/stoichiometric vectors; it does not implement SIMD system descriptors.
"""

from __future__ import annotations

from collections import defaultdict
import re

import numpy as np
from numpy.typing import NDArray

ELEMENT_SYMBOLS = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
)

ATOMIC_NUMBERS = {symbol: index for index, symbol in enumerate(ELEMENT_SYMBOLS, start=1)}
_NUMBER_PATTERN = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)")


def _normalize_formula(formula: str) -> str:
    text = str(formula).strip()
    if not text:
        raise ValueError("formula cannot be empty")
    text = re.sub(r"\s+", "", text)
    return (
        text.replace("[", "(")
        .replace("{", "(")
        .replace("]", ")")
        .replace("}", ")")
    )


class _FormulaParser:
    """Small recursive-descent parser for common inorganic formulas."""

    def __init__(self, formula: str) -> None:
        self.formula = _normalize_formula(formula)
        self.position = 0

    def parse(self) -> dict[str, float]:
        values = self._parse_group(stop_char=None)
        if self.position != len(self.formula):
            char = self.formula[self.position]
            raise ValueError(f"Unexpected character {char!r} at position {self.position}")
        if not values:
            raise ValueError("formula did not contain any elements")
        return dict(values)

    def _parse_group(self, stop_char: str | None) -> defaultdict[str, float]:
        values: defaultdict[str, float] = defaultdict(float)
        while self.position < len(self.formula):
            char = self.formula[self.position]
            if stop_char is not None and char == stop_char:
                return values
            if char == "(":
                self.position += 1
                inner = self._parse_group(stop_char=")")
                if self.position >= len(self.formula) or self.formula[self.position] != ")":
                    raise ValueError("Unclosed parenthesis in formula")
                self.position += 1
                multiplier = self._parse_number()
                for symbol, amount in inner.items():
                    values[symbol] += amount * multiplier
                continue
            if char == ")":
                if stop_char is None:
                    raise ValueError("Unmatched closing parenthesis in formula")
                return values
            if char.isupper():
                symbol = self._parse_element_symbol()
                amount = self._parse_number()
                values[symbol] += amount
                continue
            raise ValueError(f"Unexpected character {char!r} at position {self.position}")
        if stop_char is not None:
            raise ValueError("Unclosed parenthesis in formula")
        return values

    def _parse_element_symbol(self) -> str:
        start = self.position
        self.position += 1
        if self.position < len(self.formula) and self.formula[self.position].islower():
            self.position += 1
        symbol = self.formula[start : self.position]
        if symbol not in ATOMIC_NUMBERS:
            raise ValueError(f"Unknown element symbol: {symbol}")
        return symbol

    def _parse_number(self) -> float:
        match = _NUMBER_PATTERN.match(self.formula, self.position)
        if match is None:
            return 1.0
        self.position = match.end()
        value = float(match.group(0))
        if value < 0:
            raise ValueError("Negative stoichiometries are not supported")
        return value


def parse_formula(formula: str) -> dict[str, float]:
    """Parse a chemical formula into element stoichiometries.

    Parentheses and decimal stoichiometries are supported. The parser is strict
    by design: unsupported annotations such as charges or hydration dots should
    be cleaned upstream rather than silently guessed.
    """
    return _FormulaParser(formula).parse()


def composition_to_stoichiometric_vector(
    formula: str,
    *,
    max_atomic_number: int = 100,
) -> NDArray[np.float64]:
    """Encode a formula as raw stoichiometric amounts from H to max Z."""
    if not 1 <= int(max_atomic_number) <= len(ELEMENT_SYMBOLS):
        raise ValueError("max_atomic_number must be between 1 and 100")
    values = parse_formula(formula)
    vector = np.zeros(int(max_atomic_number), dtype=float)
    for symbol, amount in values.items():
        atomic_number = ATOMIC_NUMBERS[symbol]
        if atomic_number <= max_atomic_number:
            vector[atomic_number - 1] = float(amount)
    return vector


def composition_to_fraction_vector(
    formula: str,
    *,
    max_atomic_number: int = 100,
) -> NDArray[np.float64]:
    """Encode a formula as element fractions normalized to sum to one."""
    vector = composition_to_stoichiometric_vector(
        formula,
        max_atomic_number=max_atomic_number,
    )
    total = float(vector.sum())
    if total <= 0:
        raise ValueError("formula has zero total stoichiometry")
    return vector / total


def element_feature_names(*, max_atomic_number: int = 100) -> list[str]:
    """Return stable sparse element feature names."""
    if not 1 <= int(max_atomic_number) <= len(ELEMENT_SYMBOLS):
        raise ValueError("max_atomic_number must be between 1 and 100")
    return [
        f"elem_{atomic_number}_{symbol}"
        for atomic_number, symbol in enumerate(ELEMENT_SYMBOLS[:max_atomic_number], start=1)
    ]


__all__ = [
    "ATOMIC_NUMBERS",
    "ELEMENT_SYMBOLS",
    "composition_to_fraction_vector",
    "composition_to_stoichiometric_vector",
    "element_feature_names",
    "parse_formula",
]
