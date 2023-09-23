import copy
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar

import pydantic

T = TypeVar("T", covariant=True)
U = TypeVar("U")
JSON = int | float | bool | str | None | Sequence["JSON"] | Mapping[str, "JSON"]


class Bind(pydantic.BaseModel, Generic[T]):
    value: T | None = None


@dataclass
class Ok(Generic[T]):
    value: T


def to_block_id(block_id: Sequence[str]) -> str:
    return json.dumps(block_id)


def from_block_id(block_id: str) -> tuple[str, ...]:
    return tuple(json.loads(block_id))


@dataclass
class Errors:
    errors: dict[tuple[str, ...], str]

    def __init__(
        self,
        errors: dict[tuple[str, ...], str] | None = None,
    ) -> None:
        self.errors = {} if errors is None else errors

    def add(self, block_id: Sequence[str], msg: str) -> None:
        self.errors[tuple(b for b in block_id)] = msg

    def __bool__(self) -> bool:
        return bool(self.errors)

    def _to_slack_json(self) -> dict[str, Any]:
        return {to_block_id(k): v for k, v in self.errors.items()}


Parsed = Ok[T] | Errors | None
"""
A Parsed[T] is either an Ok[T] if parsing went well, some Errors if parsing
went poorly, or None if parsing was skipped (e.g. because some part of a modal
hasn't even been filled in yet).
"""


class Eventual(ABC, Generic[T]):
    """
    An Eventual[T] will "eventually be a T" once the user submits their modal:
    specifically, it's something that knows how to parse some chunk of the json
    that Slack will send us.

    Given an Eventual[T] and a function from T -> U, you can produce an
    Eventual[U] using the .map() method (Eventuals are functors).

    An Eventual[T] can also be "bound" to a Bind[T]. This is a hack, and should
    be used only in action callbacks.
    """

    _xform: Callable[[Any], T]
    _bind: Bind[T] | None
    _inner: "Eventual[Any] | None"

    def __init__(self) -> None:
        super().__init__()
        self._xform = lambda x: x
        self._bind = None
        self._inner = None

    def parse(self, payload: object) -> Parsed[T]:
        p = self._parse(payload)
        if isinstance(p, Ok):
            if self._bind is not None:
                self._bind.value = p.value
        return p

    def _parse(self, payload: object) -> Parsed[T]:
        raw = self.extract(payload)
        if not isinstance(raw, Ok):
            return raw
        return Ok(self._xform(raw.value))

    def extract(self, payload: object) -> Parsed[object]:
        if self._inner is not None:
            return self._inner.parse(payload)

        return self._extract(payload)

    @abstractmethod
    def _extract(self, payload: object) -> Parsed[object]:
        """
        Given a raw payload from Slack, extract (but don't transform) the
        "value" from it. E.g. a DatePicker would extract a YYYY-MM-DD string.
        Slack isn't consistent in where it stores these values, so each Slack
        thing has to specify their own extraction logic.
        """
        ...

    # Despite having an implementation, `map` is still an abstractmethod.
    # Subclasses should implement it by calling super(), but they should
    # give their implementation a more accurate return type:
    #
    #   def map(self, f: Callable[[T], U]) -> "MoreSpecificType[U]":
    #     u = super().map(f)
    #     assert isinstance(u, self.__class__)
    #     return u
    #
    # This trick gives a uniform way of implementing map on all subclases
    # while getting accurate types, with just a bit of boilerplate.
    @abstractmethod
    def map(self, f: Callable[[T], U]) -> "Eventual[U]":
        u = copy.copy(self)
        # Callbacks need to run with the right set of .map() transformations
        # Each call to .map can change the type that gets passed to subsequent
        # callbacks, so we keep track of a linked list of Elements, each of
        # which keeps track of a single .map()'s worth of type-appropriate
        # callbacks.
        u._inner = self
        u._xform = f  # type: ignore
        # start afresh with no callback
        return u  # type: ignore

    # Ditto.
    @abstractmethod
    def bind(self, bind: Bind[T]) -> "Eventual[T]":
        t = copy.copy(self)
        t._bind = bind
        return t


@dataclass
class Mrkdwn:
    text: str

    def _to_slack_json(self) -> Mapping[str, JSON]:
        return {"type": "mrkdwn", "text": self.text}


@dataclass
class PlainText:
    text: str
    emoji: bool = True

    def _to_slack_json(self) -> Mapping[str, JSON]:
        return {"type": "plain_text", "text": self.text, "emoji": self.emoji}


Text = Mrkdwn | PlainText


@dataclass(kw_only=True)
class Option:
    text: str
    value: str

    def to_slack_json(self) -> dict[str, Any]:
        return {"text": PlainText(self.text)._to_slack_json(), "value": self.value}

    @staticmethod
    def _from(x: "str | Option") -> "Option":
        if not isinstance(x, Option):
            x = Option(text=x, value=x)
        return x
