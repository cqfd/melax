from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar

T = TypeVar("T", covariant=True)
U = TypeVar("U")
JSON = int | float | bool | str | None | Sequence["JSON"] | Mapping[str, "JSON"]


class Mappable(ABC, Generic[T]):
    """
    Mixin to make a type "mappable", i.e. a functor.
    """

    _xform: Callable[[Any], T]

    def __init__(self) -> None:
        super().__init__()
        self._xform = lambda x: x

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
    def map(self, f: Callable[[T], U]) -> "Mappable[U]":
        u = deepcopy(self)
        u._xform = lambda x: f(self._xform(x))  # type: ignore
        return u  # type: ignore


@dataclass
class Ok(Generic[T]):
    value: T


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
