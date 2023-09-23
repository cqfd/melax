from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Literal,
    Mapping,
    Self,
    Sequence,
    TypeVar,
    overload,
)

from .elements import Element
from .types import (
    JSON,
    Bind,
    Errors,
    Eventual,
    Mrkdwn,
    Ok,
    Option,
    Parsed,
    PlainText,
    Text,
)

T = TypeVar("T", covariant=True)
U = TypeVar("U", covariant=True)


class Blocks(Eventual[T]):
    """
    A value of type Blocks[T] is a recipe for some number of blocks that will
    eventually produce a value of type T once the user submits the modal.
    """

    _name: list[str]

    def __init__(self) -> None:
        super().__init__()
        self._name = []
        self._bind = None

    @abstractmethod
    def _to_slack_blocks_json(self) -> Sequence[Mapping[str, JSON]]:
        ...

    @abstractmethod
    def _on_block_action(
        self, block_id: list[str], action_id: str, action: object
    ) -> None:
        ...

    @abstractmethod
    def _on_block_options(
        self, block_id: list[str], action_id: str, query: str
    ) -> Sequence[Option]:
        ...

    def __set_name__(self, owner: Any, name: str) -> None:
        self._name.insert(0, name)

    @abstractmethod
    def is_interactive(self) -> bool:
        ...

    # Type-specific boilerplate
    def map(self, f: Callable[[T], U]) -> "Blocks[U]":
        u = super().map(f)
        assert isinstance(u, Blocks)
        u._bind = None
        return u

    def bind(self, bind: Bind[T]) -> "Blocks[T]":
        t = super().bind(bind)
        assert isinstance(t, self.__class__)
        return t

    # Descriptor hack
    if TYPE_CHECKING:

        @overload
        def __get__(self, obj: None, objtype: Any) -> Self:
            ...

        @overload
        def __get__(self, obj: Any, objtype: Any) -> T:
            ...

        def __get__(self, obj: Any, objtype: Any) -> T | Self:
            ...


class Block(Blocks[T]):
    """
    https://api.slack.com/reference/block-kit/blocks?ref=bk
    """

    def _to_slack_blocks_json(self) -> Sequence[Mapping[str, JSON]]:
        if self._block_id is None:
            return [self._to_slack_json()]
        else:
            return [{"block_id": self._block_id} | self._to_slack_json()]

    @abstractmethod
    def _to_slack_json(self) -> Mapping[str, JSON]:
        ...

    def _on_block_action(
        self, block_id: list[str], action_id: str, action: object
    ) -> None:
        return self._on_action(action_id, action)

    @abstractmethod
    def _on_action(self, action_id: str, action: object) -> None:
        ...

    def _on_block_options(
        self, block_id: list[str], action_id: str, query: str
    ) -> Sequence[Option]:
        return self._on_options(action_id, query)

    @abstractmethod
    def _on_options(self, action_id: str, query: str) -> Sequence[Option]:
        ...

    @property
    def _block_id(self) -> str | None:
        return "$".join(self._name) if self._name else None

    # Type-specific boilerplate
    def map(self, f: Callable[[T], U]) -> "Block[U]":
        u = super().map(f)
        assert isinstance(u, self.__class__)
        return u


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
        p = cls._dict.parse(payload)
        if p is None or isinstance(p, Errors):
            return p

        instance = cls()
        for k, v in p.value.items():
            setattr(instance, k, v)
        return Ok(instance)

    @classmethod
    def _to_slack_blocks_json(cls) -> Sequence[JSON]:
        return cls._dict._to_slack_blocks_json()

    @classmethod
    def _on_block_action(
        cls, block_id: list[str], action_id: str, action: object
    ) -> None:
        return cls._dict._on_block_action(block_id, action_id, action)

    @classmethod
    def _on_block_options(
        cls, block_id: list[str], action_id: str, query: str
    ) -> Sequence[Option]:
        return cls._dict._on_block_options(block_id, action_id, query)


X = TypeVar("X", covariant=True)


class NestedBlocks(Blocks[T]):
    def __init__(
        self: "NestedBlocks[dict[str, X]]",
        blocks: Mapping[str, "Blocks[X]"],
        *,
        rename_children: bool = True,
    ) -> None:
        super().__init__()
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

    def _to_slack_blocks_json(self) -> Sequence[Mapping[str, JSON]]:
        return [
            block_json
            for block in self.blocks.values()
            for block_json in block._to_slack_blocks_json()
        ]

    def _extract(self, payload: object) -> Parsed[object]:
        assert isinstance(payload, dict)
        assert (
            self.blocks.keys() >= payload.keys()
        ), f"Unexpected keys: {payload.keys() - self.blocks.keys()}, {payload=}"

        result = {}
        errors: dict[str, str] = {}
        for name, block in self.blocks.items():
            v = payload.get(name)
            p = block.parse(payload.get(name))
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

        return Ok(result)

    def _on_block_action(
        self, block_id: list[str], action_id: str, action: object
    ) -> None:
        self.blocks[block_id[0]]._on_block_action(block_id[1:], action_id, action)

    def _on_block_options(
        self, block_id: list[str], action_id: str, query: str
    ) -> Sequence[Option]:
        return self.blocks[block_id[0]]._on_block_options(
            block_id[1:], action_id, query
        )

    def __set_name__(self, owner: Any, name: str) -> None:
        super().__set_name__(owner, name)

        for block in self.blocks.values():
            block.__set_name__(owner, name)

    def is_interactive(self) -> bool:
        return any(block.is_interactive() for block in self.blocks.values())

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


def blocks(blocks: Mapping[str, Blocks[T]]) -> Blocks[Mapping[str, T]]:
    return NestedBlocks(blocks=blocks)  # type: ignore


class Actions(Block[T]):
    def __init__(self: "Actions[dict[str, Any]]", **elements: Element[Any]) -> None:
        assert elements, "Actions blocks must have at least one element"
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

    def _extract(self, payload: object) -> Ok[object]:
        results: dict[str, Any] = {}
        for k, e in self.elements.items():
            # Confusing: the elements in an Actions block are always optional,
            # so we need to give our elements a chance to parse something even
            # if we received a totally empty payload (None) for this block.
            # This will happen e.g. if your Actions block has only buttons.
            p = e._parse(payload.get(k) if isinstance(payload, dict) else None)
            match p:
                case None:
                    results[k] = None
                case Ok(v):
                    results[k] = v

        return Ok(results)

    def _on_action(self, action_id: str, action: object) -> None:
        self.elements[action_id]._on_action(action)

    def _on_options(self, action_id: str, query: str) -> Sequence[Option]:
        return self.elements[action_id]._on_options(query)

    def is_interactive(self) -> bool:
        return any(e.is_interactive() for e in self.elements.values())

    # Type-specific boilerplate
    def map(self, f: Callable[[T], U]) -> "Actions[U]":
        u = super().map(f)
        assert isinstance(u, self.__class__)
        return u


@dataclass
class Image:
    url: str
    alt_text: str

    def _to_slack_json(self) -> Mapping[str, JSON]:
        return {"type": "image", "image_url": self.url, "alt_text": self.alt_text}


class Context(Block[T]):
    def __init__(self: "Context[None]", *elements: Text | Image) -> None:
        self.elements = elements

    if TYPE_CHECKING:

        def __new__(cls, *elements: Text | Image) -> "Context[None]":
            ...

    def _to_slack_json(self) -> Mapping[str, JSON]:
        return {
            "type": "context",
            "elements": [e._to_slack_json() for e in self.elements],
        }

    def _extract(self, payload: object) -> Ok[None]:
        return Ok(None)

    def _on_action(self, action_id: str, action: object) -> None:
        raise Exception(f"Contexts can't respond to actions: {action_id=} {action=}")

    def _on_options(self, action_id: str, query: str) -> Sequence[Option]:
        raise Exception(f"Contexts can't respond to options: {action_id=} {query=}")

    def is_interactive(self) -> bool:
        return False


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

    def is_interactive(self) -> bool:
        return False

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

    def _extract(self, payload: object) -> Parsed[object]:
        if payload is None:
            # Bit confusing: only Input blocks can have non-optional elements.
            # It's totally fine for a Section to not get anything from its
            # element.
            return Ok(None)

        assert (
            self.accessory is not None
        ), f"Section without an accessory got payload: {payload=}"
        assert isinstance(payload, dict)

        action_id = self.accessory.__class__.__name__
        p = self.accessory.parse(payload.get(action_id))
        if p is None:
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

    def _on_options(self, action_id: str, query: str) -> Sequence["Option"]:
        assert self.accessory is not None
        return self.accessory._on_options(query)

    def is_interactive(self) -> bool:
        # Section elements seem to always trigger block_action webhooks, even
        # if you haven't attached a callback.
        return self.accessory is not None

    # Type-specific boilerplate
    def map(self, f: Callable[[T], U]) -> "Section[U]":
        u = super().map(f)
        assert isinstance(u, self.__class__)
        return u


class Input(Block[T]):
    """
    https://api.slack.com/reference/block-kit/blocks?ref=bk#input

    Input blocks are special: only they can produce red error messages.
    """

    @overload
    def __init__(self: "Input[T]", label: str, element: Element[T]) -> None:
        ...

    @overload
    def __init__(
        self: "Input[T]", label: str, element: Element[T], optional: Literal[False]
    ) -> None:
        ...

    @overload
    def __init__(
        self: "Input[T | None]", label: str, element: Element[T], optional: bool
    ) -> None:
        ...

    def __init__(self, label: str, element: Element[T], optional: bool = False) -> None:
        super().__init__()
        self._xform = lambda x: Ok(x)  # type: ignore
        self.label = label
        self.element = element
        self.optional = optional

    # urgh, pyright doesn't handle self annotations the same way mypy does
    if TYPE_CHECKING:

        @overload
        def __new__(cls, label: str, element: Element[T]) -> "Input[T]":
            ...

        @overload
        def __new__(
            cls, label: str, element: Element[T], optional: Literal[False]
        ) -> "Input[T]":
            ...

        @overload
        def __new__(
            cls, label: str, element: Element[T], optional: bool
        ) -> "Input[T | None]":
            ...

        def __new__(
            cls, label: str, element: Element[T], optional: bool = False
        ) -> "Input[T | None]":
            ...

    def map_or_error_msg(self, validator: Callable[[T], Ok[U] | str]) -> "Input[U]":
        """
        Like .map() but lets you return an error message if you don't like the
        Input block's parsed value. Potentially handy if you want to "refine"
        an Input block's type in a fallible way.
        """
        return super().map(validator)  # type: ignore

    def map(self, f: Callable[[T], U]) -> "Input[U]":
        def v(t: T) -> Ok[U]:  # type: ignore
            return Ok(f(t))

        return self.map_or_error_msg(v)

    def validate(self, validator: Callable[[T], U]) -> "Input[U]":
        """
        Like map/map_or_error_msg but you can signal erros by raising
        ValueErrors.
        """

        # mypy complains about covariance but it's ok
        def v(t: T) -> Ok[U] | str:  # type: ignore
            try:
                u = validator(t)
            except ValueError as e:
                return str(e)
            else:
                return Ok(u)

        return self.map_or_error_msg(v)

    def error_if(self, condition: Callable[[T], bool], error_msg: str) -> "Input[T]":
        # mypy will complain that we're using a parameter with a covariant
        # type, but it's fine here
        def v(t: T) -> Ok[T] | str:  # type: ignore
            return Ok(t) if not condition(t) else error_msg

        return self.map_or_error_msg(v)

    def _to_slack_json(self) -> Mapping[str, JSON]:
        return {
            "type": "input",
            "label": PlainText(self.label)._to_slack_json(),
            "element": self.element._to_slack_json()
            | {"action_id": self.element.__class__.__name__},
            "optional": self.optional,
            "dispatch_action": self.element._cb is not None,
        }

    # override the default _parse implementation from Eventual
    # here's where we can finally signal errors.
    def _parse(self, payload: object) -> Parsed[T]:
        processed = super()._parse(payload)
        if processed is None or isinstance(processed, Errors):
            return processed
        assert isinstance(processed, Ok)
        if isinstance(processed.value, str):
            assert self._block_id is not None
            return Errors({self._block_id: processed.value})
        return processed.value  # type: ignore

    def _extract(self, payload: object) -> Parsed[object]:
        if payload is None:
            return Ok(None) if self.optional else None

        assert isinstance(payload, dict)
        action_id = self.element.__class__.__name__

        return self.element.parse(payload.get(action_id))

    def _on_action(self, action_id: str, action: object) -> None:
        assert action_id == self.element.__class__.__name__
        assert isinstance(action, dict)
        return self.element._on_action(action)

    def _on_options(self, action_id: str, query: str) -> Sequence[Option]:
        return self.element._on_options(query)

    def is_interactive(self) -> bool:
        return self.element.is_interactive()

    def error(self, msg: str) -> dict[str, str]:
        """
        Helper to tr to make it a little more ergonomic to construct block
        error messages.
        """
        assert self._block_id is not None
        return {self._block_id: msg}
