import datetime
import json
from abc import ABC, abstractmethod
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
import pydantic.json

T = TypeVar("T", covariant=True)
JSON = int | float | bool | str | None | Sequence["JSON"] | Mapping[str, "JSON"]

### Core types:


class Block(ABC, Generic[T]):
    """
    https://api.slack.com/reference/block-kit/blocks?ref=bk
    """

    _block_id: str

    @abstractmethod
    def _parse(self, payload: object) -> T:
        ...

    @abstractmethod
    def _to_slack_json(self) -> Mapping[str, JSON]:
        ...

    @abstractmethod
    def _on_action(self, action_id: str, action: object) -> None:
        ...

    def _on_block_action(self, block_id: str, action_id: str, action: object) -> None:
        assert isinstance(action, dict)
        return self._on_action(action_id, action)

    @abstractmethod
    def _on_options(self, action_id: str, query: str) -> list["Option"]:
        ...

    def _on_block_options(
        self, block_id: str, action_id: str, query: str
    ) -> list["Option"]:
        return self._on_options(action_id, query)

    def error(self, msg: str) -> dict[str, str]:
        return {self._block_id: msg}

    def __set_name__(self, owner: Any, name: str) -> None:
        if not hasattr(self, "_block_id"):
            self._block_id = name
        else:
            self._block_id = f"{name}${self._block_id}"

    # IntoBlocks protocol
    def _to_slack_blocks(self) -> Sequence[JSON]:
        return [self._to_slack_json() | {"block_id": self._block_id}]

    # Descriptor hack
    if TYPE_CHECKING:

        @overload
        def __get__(self, obj: "Blocks", objtype: type["Blocks"]) -> T:
            ...

        @overload
        def __get__(self, obj: None, objtype: type["Blocks"]) -> Self:
            ...

        def __get__(self, obj: "Blocks" | None, objtype: type["Blocks"]) -> T | Self:
            ...


class Element(ABC, Generic[T]):
    """
    https://api.slack.com/reference/block-kit/block-elements
    """

    _callback: Callable[[T], None] | None = None

    @property
    @abstractmethod
    def _payload_path(self) -> list[str]:
        """Path to extract the payload from an action dict (slack isn't
        consistent about this across element types)"""
        ...

    @abstractmethod
    def _parse_payload(self, payload: object) -> T:
        ...

    @abstractmethod
    def _to_slack_json(self) -> Mapping[str, JSON]:
        ...

    def _parse(self, value: object) -> T:
        payload = self._extract_payload(value)
        return self._parse_payload(payload)

    def _extract_payload(self, value: object) -> object:
        if value is None:
            return None

        assert isinstance(value, dict)
        v = value
        for k in self._payload_path:
            v = v[k]
        return v

    def _on_action(self, action: object) -> None:
        payload = self._extract_payload(action)
        value = self._parse_payload(payload)
        if self._callback is not None:
            self._callback(value)

    def _on_options(self, query: str) -> list["Option"]:
        raise Exception(f"Can't get options for element of type {self.__class__}")


class Blocks:
    """
    A DSL for building collections of Slack blocks.
    """

    _dict: ClassVar["NestedBlocks"]

    def __init_subclass__(cls) -> None:
        cls._dict = NestedBlocks(
            {
                k: v
                for k, v in cls.__dict__.items()
                if isinstance(v, Block | NestedBlocks)
            },
            rename_children=False,
        )

    @classmethod
    def _parse(cls, payload: object) -> Self:
        p = cls._dict._parse(payload)
        self = cls()
        for k, v in p.items():
            setattr(self, k, v)
        return self

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

    @classmethod
    def get(cls, block_id: str) -> Any:
        block = getattr(cls, block_id)
        assert isinstance(block, Block | NestedBlocks)
        return block

    @classmethod
    def add(
        cls,
        block_id: str,
        block: Block[Any],
        *,
        after: str | None = None,
        before: str | None = None,
    ) -> None:
        assert block_id not in cls.__dict__, f"Block {block_id} already exists"
        block.__set_name__(cls, block_id)
        setattr(cls, block_id, block)

    def __getitem__(self, block_id: str) -> Any:
        return getattr(self, block_id)

    def __contains__(self, block_id: str) -> bool:
        return hasattr(self, block_id)


class NestedBlocks:
    def __init__(
        self, blocks: Mapping[str, "IntoBlocks"], *, rename_children: bool = True
    ) -> None:
        self.blocks = blocks
        if rename_children:
            for block_id, block in blocks.items():
                block.__set_name__(self, block_id)

    def _to_slack_blocks(self) -> Sequence[JSON]:
        result: list[JSON] = []
        for block in self.blocks.values():
            if isinstance(block, Block | NestedBlocks):
                result.extend(block._to_slack_blocks())
        return result

    def _parse(self, payload: object) -> dict[str, Any]:
        assert isinstance(payload, dict)
        result = {}

        toplevel_keys = {k.split("$")[0] for k in payload}
        assert (
            self.blocks.keys() >= toplevel_keys
        ), f"Unexpected keys: {toplevel_keys - self.blocks.keys()}, {payload=}"

        for name, block in self.blocks.items():
            if isinstance(block, Block):
                v = payload.get(name)
                if v is None:
                    result[name] = block._parse(None)
                else:
                    assert isinstance(v, dict)
                    assert len(v.keys()) == 1, f"Unexpected value: {v}"
                    action_id = list(v.keys())[0]
                    result[name] = block._parse(v[action_id])
            elif isinstance(block, NestedBlocks):
                rec = block._parse(
                    {
                        k.removeprefix(f"{name}$"): v
                        for k, v in payload.items()
                        if k.startswith(f"{name}$")
                    }
                )
                result[name] = rec

        return result

    def _on_block_action(self, block_id: str, action_id: str, action: object) -> None:
        path = block_id.split("$", 1)
        # ew
        suffix = "$".join(path[1:])
        self.blocks[path[0]]._on_block_action(suffix, action_id, action)

    def _on_block_options(
        self, block_id: str, action_id: str, query: str
    ) -> list["Option"]:
        path = block_id.split("$", 1)
        # ew
        suffix = "$".join(path[1:])
        return self.blocks[path[0]]._on_block_options(suffix, action_id, query)

    def __set_name__(self, owner: Any, name: str) -> None:
        for block in self.blocks.values():
            block.__set_name__(owner, name)

    def __getitem__(self, key: str) -> Any:
        return self.blocks[key]

    # Descriptor hack
    if TYPE_CHECKING:

        @overload
        def __get__(self, obj: "Blocks", objtype: type["Blocks"]) -> dict[str, Any]:
            ...

        @overload
        def __get__(self, obj: None, objtype: type["Blocks"]) -> Self:
            ...

        def __get__(
            self, obj: "Blocks" | None, objtype: type["Blocks"]
        ) -> dict[str, Any] | Self:
            ...


IntoBlocks = Union[Block[Any], NestedBlocks]


def nested(**blocks: IntoBlocks) -> NestedBlocks:
    return NestedBlocks(blocks=blocks)


OnSubmit = Union[
    "Modal",  # continue being a (possibly different) modal
    "Errors",  # validation errors
    "Push",  # push a new modal
    None,  # finish, clearing the whole modal stack.
]
"""
Your options for what to do after the user submits a modal.
"""


Bs = TypeVar("Bs", bound=Blocks)


class View:
    """
    Bundles a collection of blocks with a submit handler, in a type-safe way.
    """

    def __init__(
        self,
        title: str,
        blocks: type[Bs],
        on_submit: tuple[(str, Callable[[Bs], "OnSubmit"])],
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


@dataclass
class Errors:
    errors: dict[str, str]

    def __init__(self, *errors: dict[str, str]) -> None:
        self.errors = {
            block_id: msg for error in errors for block_id, msg in error.items()
        }


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

    def to_slack_view_json(self) -> dict[str, Any]:
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


Text = Mrkdwn | PlainText


class Divider(Block[None]):
    def _parse(self, payload: object) -> None:
        return None

    def _to_slack_json(self) -> Mapping[str, JSON]:
        return {"type": "divider"}

    def _on_action(self, action_id: str, action: object) -> None:
        raise Exception(f"Dividers can't respond to actions: {action_id=} {action=}")

    def _on_options(self, action_id: str, query: str) -> list["Option"]:
        raise Exception(f"Dividers can't respond to options: {action_id=} {query=}")


class Section(Block[T]):
    @overload
    def __init__(self: "Section[None]", text: Text, fields: list[Text] = []) -> None:
        ...

    @overload
    def __init__(
        self: "Section[T]",
        text: Text,
        fields: list[Text] = [],
        accessory: Element[T] = ...,
    ) -> None:
        ...

    def __init__(
        self, text: Text, fields: list[Text] = [], accessory: Element[T] | None = None
    ) -> None:
        self.text = text
        self.fields = fields
        self.accessory = accessory

    def _parse(self, payload: object) -> T:
        if self.accessory is None:
            assert payload is None, f"Unexpected payload: {payload}"
            return None  # type: ignore

        return self.accessory._parse(payload)

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
                {"accessory": self.accessory._to_slack_json()} if self.accessory else {}
            ),
        }

    def _on_action(self, action_id: str, action: object) -> None:
        assert self.accessory is not None
        return self.accessory._on_action(action)

    def _on_options(self, action_id: str, query: str) -> list["Option"]:
        assert self.accessory is not None
        return self.accessory._on_options(query)


class Input(Block[T]):
    """
    https://api.slack.com/reference/block-kit/blocks?ref=bk#input
    """

    @overload
    def __init__(self: "Input[T]", label: str, element: Element[T]) -> None:
        ...

    @overload
    def __init__(
        self: "Input[T | None]", label: str, element: Element[T], optional: bool = ...
    ) -> None:
        ...

    def __init__(self, label: str, element: Element[T], optional: bool = False) -> None:
        self.label = label
        self.element = element
        self.optional = optional

    def _to_slack_json(self) -> Mapping[str, JSON]:
        return {
            "type": "input",
            "label": PlainText(self.label)._to_slack_json(),
            "element": self.element._to_slack_json()
            | {"action_id": self.element.__class__.__name__},
            "optional": self.optional,
            "dispatch_action": self.element._callback is not None,
        }

    def _parse(self, payload: object) -> T:
        if self.optional and payload is None:
            return None  # type: ignore

        return self.element._parse(payload)

    def _on_action(self, action_id: str, action: object) -> None:
        return self.element._on_action(action)

    def _on_options(self, action_id: str, query: str) -> list["Option"]:
        return self.element._on_options(query)


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
        on_click: Callable[[str], None] | None = None,
    ) -> None:
        ...

    @overload
    def __init__(
        self: "Button[None]", text: str, *, on_click: Callable[[], None] | None = None
    ) -> None:
        ...

    def __init__(
        self,
        text: str,
        *,
        value: str | None = None,
        on_click: Callable[..., None] | None = None,
    ) -> None:
        self.text = text
        if value is not None:
            assert value != "", "Slack doesn't like empty strings for button values"
        self.value = value

        def _on_click(_value: str | None) -> None:
            assert on_click is not None
            if value is not None:
                on_click(_value)
            else:
                on_click()

        self._callback = _on_click if on_click is not None else None  # type: ignore

    @property
    def _payload_path(self) -> list[str]:
        return ["value"]

    def _parse_payload(self, payload: object) -> T:
        return self.value  # type: ignore

    def _to_slack_json(self) -> Mapping[str, JSON]:
        return {
            "type": "button",
            "text": PlainText(self.text)._to_slack_json(),
            **({"value": self.value} if self.value is not None else {}),
        }


class DatePicker(Element[datetime.date]):
    """
    https://api.slack.com/reference/block-kit/block-elements#datepicker
    """

    def __init__(
        self,
        initial_date: datetime.date | None = None,
        placeholder: str | None = None,
        on_selection: Callable[[datetime.date], None] | None = None,
    ) -> None:
        self.initial_date = initial_date
        self.placeholder = placeholder
        self._callback = on_selection

    def _parse_payload(self, payload: object) -> datetime.date:
        assert isinstance(payload, str)
        return datetime.datetime.strptime(payload, "%Y-%m-%d").date()

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

    @property
    def _payload_path(self) -> list[str]:
        return ["selected_date"]


class PlainTextInput(Element[str]):
    """
    https://api.slack.com/reference/block-kit/block-elements#input
    """

    def __init__(
        self,
        initial_value: str | None = None,
        multiline: bool = False,
        focus_on_load: bool = False,
        placeholder: str | None = None,
    ) -> None:
        self.initial_value = initial_value
        self.multiline = multiline
        self.focus_on_load = focus_on_load
        self.placeholder = placeholder

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
        }

    def _parse_payload(self, payload: object) -> str:
        assert isinstance(payload, str)
        return payload

    @property
    def _payload_path(self) -> list[str]:
        return ["value"]


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
        self.is_decimal_allowed = is_decimal_allowed

    def _to_slack_json(self) -> Mapping[str, JSON]:
        return {
            "type": "number_input",
            "is_decimal_allowed": self.is_decimal_allowed,
        }

    def _parse_payload(self, payload: object) -> T:
        assert isinstance(payload, str)
        if self.is_decimal_allowed:
            return float(payload)  # type: ignore
        else:
            return int(payload)  # type: ignore

    @property
    def _payload_path(self) -> list[str]:
        return ["value"]


@dataclass
class Option:
    text: str
    value: str

    def to_slack_json(self) -> dict[str, Any]:
        return {"text": PlainText(self.text)._to_slack_json(), "value": self.value}


class Select(Element[str]):
    """
    https://api.slack.com/reference/block-kit/block-elements#select
    """

    def __init__(
        self,
        *,
        options: list[Option] | Callable[[str], list[Option]],
        placeholder: str | None = None,
        on_selection: Callable[[str], None] | None = None,
    ) -> None:
        self.options = options
        self.placeholder = placeholder
        self._callback = on_selection

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

    def _parse_payload(self, payload: object) -> str:
        assert isinstance(payload, str)
        return payload

    @property
    def _payload_path(self) -> list[str]:
        return ["selected_option", "value"]

    def _on_options(self, query: str) -> list["Option"]:
        assert callable(self.options)
        return self.options(query)
