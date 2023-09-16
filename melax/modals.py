import datetime
import json
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Literal,
    Mapping,
    Self,
    Sequence,
    TypeVar,
    Union,
    overload,
)

import pydantic

T = TypeVar("T", covariant=True)
U = TypeVar("U")
JSON = int | float | bool | str | None | Sequence["JSON"] | Mapping[str, "JSON"]

### Core types:


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


class DescriptorHack(Generic[T]):
    """
    Tell mypy how to interpret class variables inside of Builder subclasses.
    A class variable of type DescriptorHack[T] will turn into a regular T on
    instances of the Builder subclass.
    """

    if TYPE_CHECKING:

        @overload
        def __get__(self, obj: "Builder", objtype: type["Builder"]) -> T:
            ...

        @overload
        def __get__(self, obj: None, objtype: type["Builder"]) -> Self:
            ...

        def __get__(self, obj: "Builder" | None, objtype: type["Builder"]) -> T | Self:
            ...


@dataclass
class Ok(Generic[T]):
    value: T


@dataclass
class Errors:
    errors: dict[str, str]

    def __init__(self, *errors: dict[str, str]) -> None:
        self.errors = {
            block_id: msg for error in errors for block_id, msg in error.items()
        }


Parsed = Ok[T] | Errors | None
"""
A Parsed[T] is either an Ok[T] if parsing went well, some Errors if
parsing poorly, or None if parsing was skipped (e.g. because some part
of a modal hasn't even been filled in yet).
"""


class Blocks(Mappable[T], DescriptorHack[T]):
    """
    A value of type Blocks[T] is a recipe for some number of blocks that will
    eventually produce a value of type T once the user submits the modal.
    """

    @abstractmethod
    def _extract(self, payload: object) -> Parsed[object]:
        ...

    def _parse(self, payload: object) -> Parsed[T]:
        raw = self._extract(payload)
        if isinstance(raw, Ok):
            return Ok(self._xform(raw.value))
        else:
            return raw

    @abstractmethod
    def _to_slack_blocks(self) -> Sequence[JSON]:
        ...

    @abstractmethod
    def _on_block_action(self, block_id: str, action_id: str, action: object) -> None:
        ...

    @abstractmethod
    def _on_block_options(
        self, block_id: str, action_id: str, query: str
    ) -> list["Option"]:
        ...

    @abstractmethod
    def __set_name__(self, owner: Any, name: str) -> None:
        ...

    # Type-specific boilerplate
    def map(self, f: Callable[[T], U]) -> "Blocks[U]":
        u = super().map(f)
        assert isinstance(u, self.__class__)
        return u


class Block(Blocks[T]):
    """
    https://api.slack.com/reference/block-kit/blocks?ref=bk
    """

    _block_id: str | None = None

    def _to_slack_blocks(self) -> Sequence[JSON]:
        if self._block_id is None:
            return [self._to_slack_json()]
        else:
            return [{"block_id": self._block_id} | self._to_slack_json()]

    @abstractmethod
    def _to_slack_json(self) -> Mapping[str, JSON]:
        ...

    @abstractmethod
    def _extract(self, payload: object) -> Ok[object] | None:
        ...

    def _on_block_action(self, block_id: str, action_id: str, action: object) -> None:
        return self._on_action(action_id, action)

    @abstractmethod
    def _on_action(self, action_id: str, action: object) -> None:
        ...

    def _on_block_options(
        self, block_id: str, action_id: str, query: str
    ) -> list["Option"]:
        return self._on_options(action_id, query)

    @abstractmethod
    def _on_options(self, action_id: str, query: str) -> list["Option"]:
        ...

    def __set_name__(self, owner: Any, name: str) -> None:
        if self._block_id is None:
            self._block_id = name
        else:
            self._block_id = f"{name}${self._block_id}"

    # Type-specific boilerplate
    def map(self, f: Callable[[T], U]) -> "Block[U]":
        u = super().map(f)
        assert isinstance(u, self.__class__)
        return u


class Element(Mappable[T]):
    """
    https://api.slack.com/reference/block-kit/block-elements
    """

    # Elements can have action callbacks, and interspersing callbacks with maps
    # is slightly tricky. See below.
    _inner: "Element[object] | None"

    _cb: Callable[[T], None] | None

    def __init__(self) -> None:
        super().__init__()
        self._inner = None
        self._cb = None

    def _parse(self, payload: object) -> Ok[T] | None:
        raw = self._extract(payload)
        if raw is None:
            return None
        return Ok(self._xform(raw.value))

    @abstractmethod
    def _extract(self, payload: object) -> Ok[object] | None:
        ...

    @abstractmethod
    def _to_slack_json(self) -> Mapping[str, JSON]:
        ...

    def _on_options(self, query: str) -> list["Option"]:
        raise Exception(f"Can't get options for element of type {self.__class__}")

    def _on_action(self, action: object) -> None:
        if self._inner is not None:
            self._inner._on_action(action)

        p = self._parse(action)
        assert p is not None, f"Failed to parse: {action=}"
        if self._cb is not None:
            self._cb(p.value)

    # As with map above, subclasses should implement callback by calling
    # super(), but they should provide a more accurate return type.
    @abstractmethod
    def _callback(self, cb: Callable[[T], None]) -> "Element[T]":
        copy = deepcopy(self)

        def _cb(t: T) -> None:  # type: ignore
            if self._cb is not None:
                self._cb(t)
            cb(t)

        copy._cb = _cb
        return copy

    def map(self, f: Callable[[T], U]) -> "Element[U]":
        u = deepcopy(self)
        # Callbacks need to run with the right set of .map() transformations
        # Each call to .map can change the type that gets passed to subsequent
        # callbacks, so we keep track of a linked list of Elements, each of
        # which keeps track of a single .map()'s worth of callbacks.
        u._inner = self
        u._xform = lambda x: f(u._inner._xform(x))  # type: ignore
        # start afresh with no callback
        u._cb = None
        return u  # type: ignore


class Builder:
    """
    A DSL for building collections of Slack blocks.
    """

    _dict: ClassVar["NestedBlocks[Mapping[str, Any]]"]

    def __init_subclass__(cls) -> None:
        blocks: Mapping[str, Blocks[Any]] = {
            k: v for k, v in cls.__dict__.items() if isinstance(v, Blocks)
        }
        cls._dict = NestedBlocks(blocks, rename_children=False)

    @classmethod
    def _parse(cls, payload: object) -> Parsed[Self]:
        p = cls._dict._parse(payload)
        if p is None:
            return None
        if isinstance(p, Errors):
            return p

        self = cls()
        for k, v in p.value.items():
            setattr(self, k, v)
        return Ok(self)

    @classmethod
    def _to_slack_blocks(cls) -> Sequence[JSON]:
        return cls._dict._to_slack_blocks()

    @classmethod
    def _on_block_action(cls, block_id: str, action_id: str, action: object) -> None:
        return cls._dict._on_block_action(block_id, action_id, action)

    @classmethod
    def _on_block_options(
        cls, block_id: str, action_id: str, query: str
    ) -> list["Option"]:
        return cls._dict._on_block_options(block_id, action_id, query)


X = TypeVar("X", covariant=True)


class NestedBlocks(Blocks[T]):
    _name: str | None

    def __init__(
        self: "NestedBlocks[dict[str, X]]",
        blocks: Mapping[str, "Blocks[X]"],
        *,
        rename_children: bool = True,
    ) -> None:
        super().__init__()
        self._name = None
        self.blocks = blocks
        if rename_children:
            for block_id, block in blocks.items():
                block.__set_name__(self, block_id)

    if TYPE_CHECKING:

        def __new__(
            cls,
            blocks: Mapping[str, "Blocks[X]"],
            *,
            rename_children: bool = True,
        ) -> "NestedBlocks[dict[str, X]]":
            ...

    def _to_slack_blocks(self) -> Sequence[JSON]:
        result: list[JSON] = []
        for block in self.blocks.values():
            if isinstance(block, Blocks):
                result.extend(block._to_slack_blocks())
        return result

    def _extract(self, payload: object) -> Parsed[object]:
        assert isinstance(payload, dict)

        toplevel_keys = {k.split("$")[0] for k in payload}
        assert (
            self.blocks.keys() >= toplevel_keys
        ), f"Unexpected keys: {toplevel_keys - self.blocks.keys()}, {payload=}"

        result = {}
        errors: dict[str, str] = {}
        for name, block in self.blocks.items():
            if isinstance(block, Block):
                v = {k: v for k, v in payload.items() if k == name}
            else:
                v = {
                    k.removeprefix(f"{name}$"): v
                    for k, v in payload.items()
                    if k.startswith(f"{name}$")
                }
            p = block._parse(v)
            match p:
                case None:
                    print(f"Failed to parse: {name=} {block=} {v=}")
                    return None
                case Errors(es):
                    errors |= es
                case Ok(x):
                    result[name] = x

        if errors:
            return Errors(errors)

        print(f"{payload=}")
        print(f"{result=}")
        print()
        return Ok(result)

    def _on_block_action(self, block_id: str, action_id: str, action: object) -> None:
        path = block_id.split("$", 1)
        suffix = "$".join(path[1:])
        self.blocks[path[0]]._on_block_action(suffix, action_id, action)

    def _on_block_options(
        self, block_id: str, action_id: str, query: str
    ) -> list["Option"]:
        path = block_id.split("$", 1)
        suffix = "$".join(path[1:])
        return self.blocks[path[0]]._on_block_options(suffix, action_id, query)

    def __set_name__(self, owner: Any, name: str) -> None:
        if self._name is None:
            self._name = name
        else:
            self._name = f"{name}${self._name}"

        for block in self.blocks.values():
            block.__set_name__(owner, name)

    # this techincally should return Blocks[Any], or something like that,
    # but it's pretty annoying to use (all you're going to do is use it to set
    # errors)
    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return list(self.blocks.values())[key]
        return self.blocks[str(key)]

    # Type-specific boilerplate
    def map(self, f: Callable[[T], U]) -> "NestedBlocks[U]":
        u = super().map(f)
        assert isinstance(u, self.__class__)
        return u


# Combinators


def nested(**blocks: Blocks[T]) -> NestedBlocks[dict[str, T]]:
    # Not sure why mypy doesn't like this (pyright thinks it's fine)
    return NestedBlocks(blocks=blocks)  # type: ignore


T1 = TypeVar("T1", covariant=True)
T2 = TypeVar("T2", covariant=True)
T3 = TypeVar("T3", covariant=True)


@overload
def sequence() -> NestedBlocks[tuple[()]]:
    ...


@overload
def sequence(b1: tuple[str, Blocks[T1]], /) -> NestedBlocks[tuple[T1]]:
    ...


@overload
def sequence(
    b1: tuple[str, Blocks[T1]], b2: tuple[str, Blocks[T2]], /
) -> NestedBlocks[tuple[T1, T2]]:
    ...


@overload
def sequence(
    b1: tuple[str, Blocks[T1]],
    b2: tuple[str, Blocks[T2]],
    b3: tuple[str, Blocks[T3]],
    /,
) -> NestedBlocks[tuple[T1, T2, T3]]:
    ...


# etc.


@overload
def sequence(*bs: tuple[str, Blocks[T]]) -> NestedBlocks[tuple[T, ...]]:
    ...


# not sure why I need Any here :/
def sequence(*bs: Any) -> Any:
    return NestedBlocks(blocks=dict(bs)).map(lambda x: tuple(x.values()))


OnSubmit = Union[
    "Modal",  # continue being a (possibly different) modal
    "Errors",  # validation errors
    "Push",  # push a new modal
    None,  # finish, clearing the whole modal stack.
]
"""
Your options for what to do after the user submits a modal.
"""


SomeBuilderSubclass = TypeVar("SomeBuilderSubclass", bound=Builder)


class View:
    """
    Bundles a collection of blocks with a submit handler, in a type-safe way.

    https://api.slack.com/reference/surfaces/views
    """

    def __init__(
        self,
        title: str,
        blocks: type[SomeBuilderSubclass],
        on_submit: tuple[(str, Callable[[SomeBuilderSubclass], "OnSubmit"])],
    ) -> None:
        self.title = title
        self.blocks = blocks
        self.on_submit = on_submit


@dataclass
class Push:
    """
    Instruction to push a new modal onto the stack.
    """

    modal: "Modal"


_modals: dict[str, type["Modal"]] = {}


class Modal(ABC, pydantic.BaseModel):
    """
    What you'll subclass in order to define a modal.

    Modals are *stateful* and must be serializable. I'm subclassing
    pydantic.BaseModel as a hacky way to get serializability; make sure
    pydantic knows how to serialize whatever state variables you use.
    """

    @abstractmethod
    def render(self) -> View:
        ...

    def __init_subclass__(cls) -> None:
        _modals[cls.__name__] = cls

    def _to_slack_view_json(self) -> dict[str, Any]:
        view = self.render()
        return {
            "type": "modal",
            "title": PlainText(view.title)._to_slack_json(),
            "blocks": view.blocks._to_slack_blocks(),
            "submit": {"type": "plain_text", "text": view.on_submit[0]},
            "callback_id": "__melax__",
            "private_metadata": json.dumps(  # <- needs to be a string
                {"type": self.__class__.__name__, "value": self.model_dump_json()}
            ),
        }


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


Text = Union[Mrkdwn, PlainText]


class Actions(Block[T]):
    def __init__(self: "Actions[dict[str, Any]]", **elements: Element[Any]) -> None:
        super().__init__()
        self.elements = elements

    if TYPE_CHECKING:

        def __new__(cls, **elements: Element[Any]) -> "Actions[dict[str, Any]]":
            ...

    def _to_slack_json(self) -> Mapping[str, JSON]:
        return {
            "type": "actions",
            "elements": [
                {"action_id": k} | e._to_slack_json() for k, e in self.elements.items()
            ],
        }

    def _extract(self, payload: object) -> Ok[object] | None:
        assert isinstance(payload, dict)
        assert self._block_id is not None
        block_id = self._block_id.split("$")[-1]

        results: dict[str, Any] = {}
        for k, e in self.elements.items():
            p = e._parse(payload[block_id].get(k))
            match p:
                case None:
                    results[k] = None
                case Ok(v):
                    results[k] = v

        return Ok(results)

    def _on_action(self, action_id: str, action: object) -> None:
        self.elements[action_id]._on_action(action)

    def _on_options(self, action_id: str, query: str) -> list["Option"]:
        return self.elements[action_id]._on_options(query)

    # Type-specific boilerplate
    def map(self, f: Callable[[T], U]) -> "Actions[U]":
        u = super().map(f)
        assert isinstance(u, self.__class__)
        return u


class Divider(Block[T]):
    def __init__(self: "Divider[None]") -> None:
        super().__init__()

    # ugh, pyright doesn't handle self annotations the same way mypy does
    if TYPE_CHECKING:

        def __new__(cls) -> "Divider[None]":
            ...

    def _extract(self, payload: object) -> Ok[None]:
        return Ok(None)

    def _to_slack_json(self) -> Mapping[str, JSON]:
        return {"type": "divider"}

    def _on_action(self, action_id: str, action: object) -> None:
        raise Exception(f"Dividers can't respond to actions: {action_id=} {action=}")

    def _on_options(self, action_id: str, query: str) -> list["Option"]:
        raise Exception(f"Dividers can't respond to options: {action_id=} {query=}")

    # Type-specific boilerplate
    def map(self, f: Callable[[T], U]) -> "Divider[U]":
        u = super().map(f)
        assert isinstance(u, self.__class__)
        return u


class Section(Block[T]):
    """
    https://api.slack.com/reference/block-kit/blocks#section
    """

    @overload
    def __init__(
        self: "Section[None]", text: str | Text, fields: list[Text] = []
    ) -> None:
        ...

    @overload
    def __init__(
        self: "Section[T | None]",
        text: str | Text,
        fields: list[Text] = [],
        accessory: Element[T] = ...,
    ) -> None:
        ...

    def __init__(
        self,
        text: str | Text,
        fields: list[Text] = [],
        accessory: Element[T] | None = None,
    ) -> None:
        super().__init__()
        self.text = Mrkdwn(text) if isinstance(text, str) else text
        self.fields = fields
        self.accessory = accessory

    # ugh, pyright doesn't handle self annotations the same way mypy does
    if TYPE_CHECKING:

        @overload
        def __new__(cls, text: str | Text, fields: list[Text] = []) -> "Section[None]":
            ...

        @overload
        def __new__(
            cls, text: str | Text, fields: list[Text] = [], accessory: Element[T] = ...
        ) -> "Section[T | None]":
            ...

        def __new__(
            cls, text: Any, fields: Any = [], accessory: Any = None
        ) -> "Section[T | None]":
            ...

    def _extract(self, payload: object) -> Ok[object] | None:
        if self.accessory is None:
            assert payload == {}, f"Unexpected payload: {payload}"
            return Ok(None)

        assert self._block_id is not None
        local_block_id = self._block_id.split("$")[-1]
        assert isinstance(payload, dict)
        if local_block_id not in payload:
            return Ok(None)

        action_id = self.accessory.__class__.__name__

        p = self.accessory._parse(payload[local_block_id][action_id])
        if p is None:
            # Bit confusing: only Input blocks can have non-optional elements.
            # It's totally fine for a Section to not get anything from its
            # element.
            return Ok(None)
        return p

    def _to_slack_json(self) -> Mapping[str, JSON]:
        return {
            "type": "section",
            "text": self.text._to_slack_json(),
            **(
                {"fields": [f._to_slack_json() for f in self.fields]}
                if self.fields
                else {}
            ),
            **(
                {
                    "accessory": {"action_id": self.accessory.__class__.__name__}
                    | self.accessory._to_slack_json()
                }
                if self.accessory
                else {}
            ),
        }

    def _on_action(self, action_id: str, action: object) -> None:
        assert self.accessory is not None
        return self.accessory._on_action(action)

    def _on_options(self, action_id: str, query: str) -> list["Option"]:
        assert self.accessory is not None
        return self.accessory._on_options(query)

    # Type-specific boilerplate
    def map(self, f: Callable[[T], U]) -> "Section[U]":
        u = super().map(f)
        assert isinstance(u, self.__class__)
        return u


class Input(Block[T]):
    """
    https://api.slack.com/reference/block-kit/blocks?ref=bk#input

    Input blocks are special: only they can have red error messages.
    """

    _validator: Callable[[object], Ok[T] | str] | None

    @overload
    def __init__(self: "Input[T]", label: str, element: Element[T]) -> None:
        ...

    @overload
    def __init__(
        self: "Input[T | None]", label: str, element: Element[T], optional: bool = ...
    ) -> None:
        ...

    def __init__(self, label: str, element: Element[T], optional: bool = False) -> None:
        super().__init__()
        self._validator = None
        self.label = label
        self.element = element
        self.optional = optional

    def map_or_error_msg(self, validator: Callable[[T], Ok[U] | str]) -> "Input[U]":
        copy = deepcopy(self)

        # mypy will complain that we're using a parameter with a covariant
        # type, but it's fine here
        def v(t: T) -> Ok[U] | str:  # type: ignore
            if self._validator is not None:
                v = self._validator(t)
                match v:
                    case Ok(t):
                        return validator(t)
                    case error_msg:
                        return error_msg
            return validator(t)

        copy._validator = v  # type: ignore
        return copy  # type: ignore

    def error_if(self, condition: Callable[[T], bool], error_msg: str) -> "Input[T]":
        # mypy will complain that we're using a parameter with a covariant
        # type, but it's fine here
        def v(t: T) -> Ok[T] | str:  # type: ignore
            return Ok(t) if not condition(t) else error_msg

        return self.map_or_error_msg(v)

    def error_unless(
        self, condition: Callable[[T], bool], error_msg: str
    ) -> "Input[T]":
        # again, covariance issue is fine here
        return self.error_if(lambda t: not condition(t), error_msg)  # type: ignore

    def _to_slack_json(self) -> Mapping[str, JSON]:
        return {
            "type": "input",
            "label": PlainText(self.label)._to_slack_json(),
            "element": self.element._to_slack_json()
            | {"action_id": self.element.__class__.__name__},
            "optional": self.optional,
            "dispatch_action": self.element._cb is not None,
        }

    def _parse(self, payload: object) -> Parsed[T]:
        raw = self._extract(payload)
        if raw is None:
            return None
        if isinstance(raw, Errors):
            return raw
        processed = self._xform(raw.value)
        if self._validator is None:
            return Ok(processed)
        validated = self._validator(processed)
        if isinstance(validated, str):
            assert self._block_id is not None
            return Errors({self._block_id: validated})
        return validated

    def _extract(self, payload: object) -> Ok[object] | None:
        assert self._block_id is not None
        local_block_id = self._block_id.split("$")[-1]
        assert isinstance(payload, dict)
        if local_block_id not in payload:
            return Ok(None) if self.optional else None

        action_id = self.element.__class__.__name__

        return self.element._parse(payload[local_block_id][action_id])

    def _on_action(self, action_id: str, action: object) -> None:
        assert action_id == self.element.__class__.__name__
        assert isinstance(action, dict)
        return self.element._on_action(action)

    def _on_options(self, action_id: str, query: str) -> list["Option"]:
        return self.element._on_options(query)

    def error(self, msg: str) -> dict[str, str]:
        """
        Helper to tr to make it a little more ergonomic to construct block
        error messages.
        """
        assert self._block_id is not None
        return {self._block_id: msg}

    # Type-specific boilerplate
    def map(self, f: Callable[[T], U]) -> "Input[U]":
        u = super().map(f)
        assert isinstance(u, self.__class__)
        return u


class Button(Element[T]):
    """
    https://api.slack.com/reference/block-kit/block-elements#button
    """

    @overload
    def __init__(
        self: "Button[str]",
        text: str,
        *,
        value: str,
        style: Literal["primary", "danger"] | None = None,
    ) -> None:
        ...

    @overload
    def __init__(
        self: "Button[None]",
        text: str,
        *,
        style: Literal["primary", "danger"] | None = None,
    ) -> None:
        ...

    def __init__(
        self,
        text: str,
        *,
        value: str | None = None,
        style: Literal["primary", "danger"] | None = None,
    ) -> None:
        super().__init__()
        self.text = text
        if value is not None:
            assert value != "", "Slack doesn't like empty strings for button values"
        self.value = value
        self.style = style

    def _parse_payload(self, payload: object) -> str | None:
        return self.value

    def _to_slack_json(self) -> Mapping[str, JSON]:
        return {
            "type": "button",
            "text": PlainText(self.text)._to_slack_json(),
            **({"value": self.value} if self.value is not None else {}),
            **({"style": self.style} if self.style is not None else {}),
        }

    def _extract(self, payload: object) -> Ok[str | None] | None:
        if self.value is not None:
            return Ok(self.value)
        return Ok(None)

    def map(self, f: Callable[[T], U]) -> "Button[U]":
        u = super().map(f)
        assert isinstance(u, self.__class__)
        return u

    def on_pressed(self, cb: Callable[[T], None]) -> "Button[T]":
        return self._callback(cb)

    def _callback(self, cb: Callable[[T], None]) -> "Button[T]":
        u = super()._callback(cb)
        assert isinstance(u, self.__class__)
        return u


class DatePicker(Element[T]):
    """
    https://api.slack.com/reference/block-kit/block-elements#datepicker
    """

    def __init__(
        self: "DatePicker[datetime.date]",
        initial_date: datetime.date | None = None,
        placeholder: str | None = None,
    ) -> None:
        super().__init__()
        self._xform = lambda s: datetime.datetime.strptime(s, "%Y-%m-%d").date()
        self.initial_date = initial_date
        self.placeholder = placeholder

    # ugh, pyright doesn't handle self annotations the same way mypy does
    if TYPE_CHECKING:

        def __new__(
            cls,
            initial_date: datetime.date | None = None,
            placeholder: str | None = None,
        ) -> "DatePicker[datetime.date]":
            ...

    def _extract(self, payload: object) -> Ok[str] | None:
        if payload is None:
            return None

        assert isinstance(payload, dict)
        v = payload["selected_date"]
        if v is None:
            return None
        assert isinstance(v, str)
        return Ok(v)

    def _to_slack_json(self) -> Mapping[str, JSON]:
        return {
            "type": "datepicker",
            **(
                {"initial_date": self.initial_date.isoformat()}
                if self.initial_date is not None
                else {}
            ),
            **(
                {"placeholder": PlainText(self.placeholder)._to_slack_json()}
                if self.placeholder is not None
                else {}
            ),
        }

    # Type-specific boilerplate
    def map(self, f: Callable[[T], U]) -> "DatePicker[U]":
        u = super().map(f)
        assert isinstance(u, self.__class__)
        return u

    def on_picked(self, cb: Callable[[T], None]) -> "DatePicker[T]":
        return self._callback(cb)

    def _callback(self, cb: Callable[[T], None]) -> "DatePicker[T]":
        u = super()._callback(cb)
        assert isinstance(u, self.__class__)
        return u


class PlainTextInput(Element[T]):
    """
    https://api.slack.com/reference/block-kit/block-elements#input
    """

    trigger_actions_on: str | None

    def __init__(
        self: "PlainTextInput[str]",
        initial_value: str | None = None,
        multiline: bool = False,
        focus_on_load: bool = False,
        placeholder: str | None = None,
    ) -> None:
        super().__init__()
        self.initial_value = initial_value
        self.multiline = multiline
        self.focus_on_load = focus_on_load
        self.placeholder = placeholder
        self.trigger_actions_on = None

    # ugh, pyright doesn't handle self annotations the same way mypy does
    if TYPE_CHECKING:

        def __new__(
            cls,
            initial_value: str | None = None,
            multiline: bool = False,
            focus_on_load: bool = False,
            placeholder: str | None = None,
        ) -> "PlainTextInput[str]":
            ...

    def _extract(self, payload: object) -> Ok[str] | None:
        print(f"PlainTextInput {payload=}")
        if payload == {}:
            return None
        assert isinstance(payload, dict), f"Unexpected payload: {payload=}"
        return Ok(payload["value"])

    def _to_slack_json(self) -> Mapping[str, JSON]:
        return {
            "type": "plain_text_input",
            "multiline": self.multiline,
            "focus_on_load": self.focus_on_load,
            **(
                {"placehodler": PlainText(self.placeholder)._to_slack_json()}
                if self.placeholder is not None
                else {}
            ),
            **(
                {
                    "dispatch_action_config": {
                        "trigger_actions_on": [self.trigger_actions_on]
                    }
                }
                if self.trigger_actions_on is not None
                else {}
            ),
        }

    def _parse_payload(self, payload: object) -> str:
        assert isinstance(payload, str)
        return payload

    def map(self, f: Callable[[T], U]) -> "PlainTextInput[U]":
        u = super().map(f)
        assert isinstance(u, self.__class__)
        return u

    def on_enter_pressed(self, cb: Callable[[T], None]) -> "PlainTextInput[T]":
        """
        https://api.slack.com/reference/block-kit/composition-objects#dispatch_action_config
        """
        assert (
            self.trigger_actions_on != "on_character_entered"
        ), f"You can only register one type of action callback for a PlainTextInput"
        copy = self._callback(cb)
        copy.trigger_actions_on = "on_enter_pressed"
        return copy

    def on_character_entered(self, cb: Callable[[T], None]) -> "PlainTextInput[T]":
        """
        https://api.slack.com/reference/block-kit/composition-objects#dispatch_action_config
        """
        assert (
            self.trigger_actions_on != "on_enter_pressed"
        ), f"You can only register one type of action callback for a PlainTextInput"
        copy = self._callback(cb)
        copy.trigger_actions_on = "on_character_entered"
        return copy

    def _callback(self, cb: Callable[[T], None]) -> "PlainTextInput[T]":
        u = super()._callback(cb)
        assert isinstance(u, self.__class__)
        return u


class NumberInput(Element[T]):
    """
    https://api.slack.com/reference/block-kit/block-elements#number
    """

    @overload
    def __init__(
        self: "NumberInput[float]", *, is_decimal_allowed: Literal[True]
    ) -> None:
        ...

    @overload
    def __init__(
        self: "NumberInput[int]", *, is_decimal_allowed: Literal[False]
    ) -> None:
        ...

    def __init__(self: "NumberInput[float | int]", *, is_decimal_allowed: bool) -> None:
        super().__init__()
        self.is_decimal_allowed = is_decimal_allowed
        self._xform = float if is_decimal_allowed else int

    def _to_slack_json(self) -> Mapping[str, JSON]:
        return {
            "type": "number_input",
            "is_decimal_allowed": self.is_decimal_allowed,
        }

    def _extract(self, payload: object) -> Ok[str] | None:
        if payload is None:
            return None
        assert isinstance(payload, dict), f"Unexpected payload: {payload=}"
        v = payload.get("value")
        assert isinstance(v, str)
        return Ok(v)

    def map(self, f: Callable[[T], U]) -> "NumberInput[U]":
        u = super().map(f)
        assert isinstance(u, self.__class__)
        return u

    def _callback(self, cb: Callable[[T], None]) -> "NumberInput[T]":
        u = super()._callback(cb)
        assert isinstance(u, self.__class__)
        return u


@dataclass
class Option:
    text: str
    value: str

    def to_slack_json(self) -> dict[str, Any]:
        return {"text": PlainText(self.text)._to_slack_json(), "value": self.value}


class Select(Element[T]):
    """
    https://api.slack.com/reference/block-kit/block-elements#select
    """

    def __init__(
        self: "Select[str]",
        *,
        options: list[Option] | Callable[[str], list[Option]],
        placeholder: str | None = None,
    ) -> None:
        super().__init__()
        self.options = options
        self.placeholder = placeholder

    # ugh, pyright doesn't handle self annotations the same way mypy does
    if TYPE_CHECKING:

        def __new__(
            cls,
            *,
            options: list[Option] | Callable[[str], list[Option]],
            placeholder: str | None = None,
        ) -> "Select[str]":
            ...

    def _to_slack_json(self) -> Mapping[str, JSON]:
        return {
            "type": "static_select"
            if isinstance(self.options, list)
            else "external_select",
            **(
                {"placeholder": PlainText(self.placeholder)._to_slack_json()}
                if self.placeholder is not None
                else {}
            ),
            **(
                {
                    "options": [
                        {"text": PlainText(o.text)._to_slack_json(), "value": o.value}
                        for o in self.options
                    ]
                }
                if isinstance(self.options, list)
                else {}
            ),
        }

    def _extract(self, payload: object) -> Ok[str] | None:
        if payload is None:
            return None

        assert isinstance(payload, dict)
        v = payload["selected_option"]["value"]
        assert isinstance(v, str)
        return Ok(v)

    def _on_options(self, query: str) -> list["Option"]:
        assert callable(self.options)
        return self.options(query)

    def map(self, f: Callable[[T], U]) -> "Select[U]":
        u = super().map(f)
        assert isinstance(u, self.__class__)
        return u

    def on_selected(self, cb: Callable[[T], None]) -> "Select[T]":
        return self._callback(cb)

    def _callback(self, cb: Callable[[T], None]) -> "Select[T]":
        u = super()._callback(cb)
        assert isinstance(u, self.__class__)
        return u
