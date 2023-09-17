import datetime
from abc import abstractmethod
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    Mapping,
    Sequence,
    TypeAlias,
    TypeVar,
    overload,
)

from .types import JSON, Mappable, Ok, Option, PlainText

T = TypeVar("T", covariant=True)
U = TypeVar("U", covariant=True)


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

    def is_interactive(self) -> bool:
        return self._cb is not None

    def _on_options(self, query: str) -> Sequence["Option"]:
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

    @abstractmethod
    def map(self, f: Callable[[T], U]) -> "Element[U]":
        u = deepcopy(self)
        # Callbacks need to run with the right set of .map() transformations
        # Each call to .map can change the type that gets passed to subsequent
        # callbacks, so we keep track of a linked list of Elements, each of
        # which keeps track of a single .map()'s worth of type-appropriate
        # callbacks.
        u._inner = self
        u._xform = lambda x: f(u._inner._xform(x))  # type: ignore
        # start afresh with no callback
        u._cb = None
        return u  # type: ignore


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

    def _to_slack_json(self) -> Mapping[str, JSON]:
        return {
            "type": "button",
            "text": PlainText(self.text)._to_slack_json(),
            **({"value": self.value} if self.value is not None else {}),
            **({"style": self.style} if self.style is not None else {}),
        }

    def _extract(self, payload: object) -> Ok[str | None]:
        # Confusing: Slack treats buttons differently between submit callbacks
        # and action callbacks; button action callbacks get a "value" field,
        # but submit callbacks *never* get button-related payloads. To fit
        # Buttons into this functor-y framework, I just pretend that they
        # always have their value.
        return Ok(self.value)

    def is_is_interactive(self) -> bool:
        return True

    def map(self, f: Callable[[T], U]) -> "Button[U]":
        u = super().map(f)
        assert isinstance(u, self.__class__)
        return u

    def on_pressed(self, cb: Callable[[T], None]) -> "Button[T]":
        return self._callback(cb)

    def _callback(self, cb: Callable[[T], None]) -> "Button[T]":
        t = super()._callback(cb)
        assert isinstance(t, self.__class__)
        return t


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
        t = super()._callback(cb)
        assert isinstance(t, self.__class__)
        return t


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
        if payload is None:
            return None
        assert isinstance(
            payload, dict
        ), f"Unexpected PlainTextInput payload: {payload=}"
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
        ), "You can only register one type of action callback for a PlainTextInput"
        copy = self._callback(cb)
        copy.trigger_actions_on = "on_enter_pressed"
        return copy

    def on_character_entered(self, cb: Callable[[T], None]) -> "PlainTextInput[T]":
        """
        https://api.slack.com/reference/block-kit/composition-objects#dispatch_action_config
        """
        assert (
            self.trigger_actions_on != "on_enter_pressed"
        ), "You can only register one type of action callback for a PlainTextInput"
        copy = self._callback(cb)
        copy.trigger_actions_on = "on_character_entered"
        return copy

    def _callback(self, cb: Callable[[T], None]) -> "PlainTextInput[T]":
        t = super()._callback(cb)
        assert isinstance(t, self.__class__)
        return t


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
        t = super()._callback(cb)
        assert isinstance(t, self.__class__)
        return t


class Select(Element[T]):
    """
    https://api.slack.com/reference/block-kit/block-elements#select
    """

    options: Sequence[Option] | Callable[[str], Sequence[Option]]

    Option: TypeAlias = Option

    def __init__(
        self: "Select[str]",
        *,
        options: Sequence[Option | str] | Callable[[str], Sequence[Option | str]],
        placeholder: str | None = None,
    ) -> None:
        """
        Providing a static list of options produces an element of type
        "static_select". To produce an element of type "external_select", pass
        a function that takes the current typeahead text as an argument and
        *returns* a list of options.
        """

        super().__init__()
        if isinstance(options, Sequence):
            self.options = [Option._from(o) for o in options]
        else:
            # not sure why this helps out mypy
            f = options
            self.options = lambda query: [Option._from(o) for o in f(query)]
        self.placeholder = placeholder

    # ugh, pyright doesn't handle self annotations the same way mypy does
    if TYPE_CHECKING:

        def __new__(
            cls,
            *,
            options: Sequence[Option | str] | Callable[[str], Sequence[Option | str]],
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

    def _on_options(self, query: str) -> Sequence["Option"]:
        assert callable(self.options)
        return self.options(query)

    def is_interactive(self) -> bool:
        return super().is_interactive() or callable(self.options)

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


class UsersSelect(Element[T]):
    def __init__(
        self: "UsersSelect[str]",
        *,
        initial_user_id: str | None = None,
        focus_on_load: bool = False,
        placeholder: str | None = None,
    ) -> None:
        super().__init__()
        self.initial_user_id = initial_user_id
        self.focus_on_load = focus_on_load
        self.placeholder = placeholder

    if TYPE_CHECKING:

        def __new__(
            cls,
            initial_user_id: str | None = None,
            focus_on_load: bool = False,
            placeholder: str | None = None,
        ) -> "UsersSelect[str]":
            ...

    def _to_slack_json(self) -> Mapping[str, JSON]:
        return {
            "type": "users_select",
            **(
                {"initial_user": self.initial_user_id}
                if self.initial_user_id is not None
                else {}
            ),
            **(
                {"placeholder": PlainText(self.placeholder)._to_slack_json()}
                if self.placeholder is not None
                else {}
            ),
        }

    def _extract(self, payload: object) -> Ok[str] | None:
        print(f"UsersSelect {payload=}")
        assert isinstance(payload, dict)
        selected_user = payload.get("selected_user")
        if selected_user is None:
            return None
        assert isinstance(selected_user, str)
        return Ok(selected_user)

    def on_selected(self, cb: Callable[[T], None]) -> "UsersSelect[T]":
        return self._callback(cb)

    def map(self, f: Callable[[T], U]) -> "UsersSelect[U]":
        u = super().map(f)
        assert isinstance(u, self.__class__)
        return u

    def _callback(self, cb: Callable[[T], None]) -> "UsersSelect[T]":
        t = super()._callback(cb)
        assert isinstance(t, self.__class__)
        return t
