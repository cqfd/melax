import datetime
from dataclasses import dataclass
from typing import Iterator, Protocol
from unittest import mock
from unittest.mock import Mock

import pytest

from melax.blocks import Actions, Builder, Errors, Input, Ok, Section, blocks
from melax.elements import (
    Button,
    DatePicker,
    NumberInput,
    PlainTextInput,
    Select,
    UsersSelect,
)
from melax.modals import Modal, OnSubmit, View
from melax.types import Bind, Option


@pytest.fixture(autouse=True)
def unregister_modals() -> Iterator[None]:
    yield
    from melax.modals import _modals

    _modals.clear()


def test_input_blocks() -> None:
    cb = Mock()

    class Form(Builder):
        name = Input("Name:", PlainTextInput().on_enter_pressed(cb))

    p = Form._parse({})
    assert p is None

    p = Form._parse({"name": {"PlainTextInput": {"value": "Foo"}}})

    assert isinstance(p, Ok)
    assert isinstance(p.value, Form)
    assert p.value.name == "Foo"

    Form._on_block_action(["name"], "PlainTextInput", {"value": "Foo"})
    cb.assert_called_with("Foo")


def test_section_blocks() -> None:
    class Form(Builder):
        s = Section("Hello!")

    p = Form._parse({})
    assert isinstance(p, Ok)
    assert isinstance(p.value, Form)
    assert p.value.s is None

    class Form2(Builder):
        s = Section("Hello", accessory=DatePicker())

    p2 = Form2._parse({})
    # Section accessory elements are always optional
    assert isinstance(p2, Ok)
    assert isinstance(p2.value, Form2)
    assert p2.value.s is None

    p2 = Form2._parse(
        {"s": {"DatePicker": {"type": "datepicker", "selected_date": None}}}
    )

    assert isinstance(p2, Ok)
    assert isinstance(p2.value, Form2)

    p2 = Form2._parse(
        {"s": {"DatePicker": {"type": "datepicker", "selected_date": "2023-09-02"}}}
    )
    assert isinstance(p2, Ok)
    assert isinstance(p2.value, Form2)
    assert p2.value.s == datetime.date.fromisoformat("2023-09-02")


def test_button_callbacks() -> None:
    cb = Mock()
    b = Button("Click me!", value="ok").map(len).on_pressed(cb)
    b._on_action({})

    cb.assert_called_with(len("ok"))


def test_button_callbacks_interspersed_with_maps() -> None:
    cb = Mock()

    b = (
        Button("Click me!", value="ok")
        .on_pressed(cb)  # <- should receive a str
        .map(len)
        .on_pressed(cb)  # <- should receive an int
        .map(lambda n: -n)
        .on_pressed(cb)
    )
    b._on_action({})

    cb.assert_has_calls([mock.call("ok"), mock.call(len("ok")), mock.call(-len("ok"))])


def test_error_helpers() -> None:
    @dataclass
    class PositiveNumber:
        n: int

        def __post_init__(self) -> None:
            assert self.n > 0

    def validate_positive(n: str) -> PositiveNumber:
        _n = int(n)
        if _n <= 0:
            raise ValueError("Must be positive")
        return PositiveNumber(_n)

    class Form(Builder):
        i = Input("Enter a number", NumberInput(is_decimal_allowed=False)).error_if(
            lambda x: x == 7, "7 is unlucky"
        )
        j = Input("Enter a positive number", PlainTextInput()).validate(
            validate_positive
        )

    p = Form._parse(
        {"i": {"NumberInput": {"value": "7"}}, "j": {"PlainTextInput": {"value": "0"}}}
    )
    assert p == Errors({("i",): "7 is unlucky", ("j",): "Must be positive"})

    p2 = Form._parse(
        {
            "i": {"NumberInput": {"value": "123"}},
            "j": {"PlainTextInput": {"value": "2"}},
        }
    )
    assert isinstance(p2, Ok)
    assert p2.value.i == 123
    assert p2.value.j == PositiveNumber(2)


def test_errors_in_submit_handlers() -> None:
    class Example(Modal):
        def render(self) -> View:
            class Form(Builder):
                i = Input("Enter a number", NumberInput(is_decimal_allowed=False))
                jk = blocks(
                    {
                        "j": Input("Enter your name", PlainTextInput()),
                        "k": Input(
                            "Enter your age", NumberInput(is_decimal_allowed=False)
                        ),
                    }
                )

            def on_submit(f: Form) -> Errors | None:
                errors = Errors()
                if f.i == 7:
                    errors.add("i", "wrong number")
                if f.jk["k"] == 100:
                    errors.add(["jk", "k"], "too old")
                if errors:
                    return errors

                return None

            return View(title="Example", builder=Form, on_submit=("Submit", on_submit))

    v = Example().render()
    p = v.builder._parse(
        {
            "i": {"NumberInput": {"value": "7"}},
            "jk": {
                "j": {"PlainTextInput": {"value": "Joe"}},
                "k": {"NumberInput": {"value": "100"}},
            },
        }
    )
    assert isinstance(p, Ok)
    assert v.on_submit[1](p.value) == Errors(
        {
            ("i",): "wrong number",
            (
                "jk",
                "k",
            ): "too old",
        }
    )


def test_actions_blocks() -> None:
    yay = Mock()
    nay = Mock()
    dob = Mock()

    class Form(Builder):
        yay_or_nay = Actions(
            yay=Button("Yay", value="yay", style="primary").on_pressed(yay),
            nay=Button("Nay", value="nay", style="danger").on_pressed(nay),
            dob=DatePicker().on_picked(dob),
        )

    yay_action = {
        "action_id": "yay",
        "block_id": "yay_or_nay",
        "text": {"type": "plain_text", "text": "Yay", "emoji": True},
        "value": "yay",
        "style": "primary",
        "type": "button",
        "action_ts": "1694859764.558458",
    }

    Form._on_block_action(["yay_or_nay"], "yay", yay_action)
    yay.assert_called_with("yay")

    today = datetime.date.today()
    today_ymd = today.strftime("%Y-%m-%d")

    date_picker_action = {
        "type": "datepicker",
        "action_id": "dob",
        "block_id": "yay_or_nay",
        "selected_date": today_ymd,
        "action_ts": "1694860134.517951",
    }

    Form._on_block_action(["yay_or_nay"], "dob", date_picker_action)
    dob.assert_called_with(today)

    submit_payload = {
        "yay_or_nay": {"dob": {"type": "datepicker", "selected_date": None}}
    }

    p = Form._parse(submit_payload)
    assert isinstance(p, Ok)
    assert p.value.yay_or_nay == {"yay": "yay", "nay": "nay", "dob": None}

    # Hmm, this is maybe a bit weird. The elemnts in an Actions block are
    # always optional. But Buttons in this framework effectively always have a
    # value, because Slack *never* sends a submit payload for them--from
    # Slack's perspective their "value" parameter only exists when they get
    # clicked. At any rate, to fit into this functor-y framework, I pretend
    # that Buttons always have their value; that means that this Actions block
    # needs to successfully parse even a totally empty payload.
    p2 = Form._parse({})
    assert isinstance(p2, Ok)
    assert p2.value.yay_or_nay == {"yay": "yay", "nay": "nay", "dob": None}


def test_select_elements() -> None:
    chocolate = Select.Option(text="Chocolate", value="chocolate")
    vanilla = Select.Option(text="Vanilla", value="vanilla")
    static_options = [chocolate, vanilla, "strawberry"]

    class Static(Builder):
        fav_ice_cream = Input("Favorite ice cream", Select(options=static_options))

    p = Static._parse(
        {
            "fav_ice_cream": {
                "Select": {
                    "type": "static_select",
                    "selected_option": {
                        "text": {
                            "type": "plain_text",
                            "text": "Chocolate",
                            "emoji": True,
                        },
                        "value": "chocolate",
                    },
                }
            }
        }
    )
    assert isinstance(p, Ok)
    assert p.value.fav_ice_cream == "chocolate"

    def external_options(query: str) -> list[Option]:
        opts = [Option._from(o) for o in static_options]
        return [o for o in opts if o.text.lower().startswith(query)]

    mock = Mock(side_effect=external_options)

    class Dynamic(Builder):
        fav_ice_cream = Input("Favorite ice cream", Select(options=mock))

    opts = Dynamic._on_block_options(["fav_ice_cream"], "Select", "choco")
    mock.assert_called_with("choco")
    assert opts == [chocolate]


def test_users_select_elements() -> None:
    cb = Mock()

    class Form(Builder):
        manager = Input("Manager", UsersSelect())
        best_friend = Section("Best friend", accessory=UsersSelect().on_selected(cb))

    p = Form._parse(
        {
            "manager": {
                "UsersSelect": {
                    "type": "users_select",
                    "selected_user": "U123",
                }
            },
            "best_friend": {
                "UsersSelect": {
                    "type": "users_select",
                    "selected_user": None,
                }
            },
        }
    )

    assert isinstance(p, Ok)
    assert p.value.manager == "U123"
    assert p.value.best_friend is None

    user_id = "U01RCM23TT2"
    action = {
        "type": "users_select",
        "action_id": "UsersSelect",
        "block_id": "best_friend",
        "selected_user": user_id,
        "action_ts": "1694925631.236937",
    }

    Form._on_block_action(["best_friend"], "UsersSelect", action)
    cb.assert_called_with(user_id)


def test_nested_parsing() -> None:
    class Form(Builder):
        stuff = blocks(
            {"A": Input("A", PlainTextInput()), "B": Input("B", PlainTextInput())}
        )

    p = Form._parse(
        {
            "stuff": {
                "A": {"PlainTextInput": {"value": None}},
                "B": {"PlainTextInput": {"value": "bar"}},
            }
        }
    )
    assert p is None


def test_binds() -> None:
    submit_cb = Mock()

    watched_x = Bind[str]()
    x = Input("X", PlainTextInput()).bind(watched_x)
    y = x.map(len)

    assert y._inner is x

    p = y.parse({"PlainTextInput": {"value": "foo"}})
    assert isinstance(p, Ok)
    assert p.value == 3

    class SomeModal(Modal):
        bound_x: Bind[str] = Bind()
        bound_int: Bind[int] = Bind()
        bound_element: Bind[str] = Bind()

        def render(self) -> View:
            class Form(Builder):
                x = (
                    Input("X", PlainTextInput().bind(self.bound_element))
                    .bind(self.bound_x)
                    .map(len)
                    .map(lambda i: -i)
                    .bind(self.bound_int)
                )

            return View(
                title="Some modal", builder=Form, on_submit=("Submit", submit_cb)
            )

    modal = SomeModal()

    b = modal.render().builder
    p = b._parse({"x": {"PlainTextInput": {"value": "foo"}}})

    assert isinstance(p, Ok)
    assert p.value.x == -3  # type: ignore

    assert modal.bound_x.value == "foo"
    assert modal.bound_int.value == -3
    assert modal.bound_element.value == "foo"


def test_weird_compositions() -> None:
    # if you're going to make a compositional library I guess you have to
    # support some weird stuff
    pass


def test_protocol_trick() -> None:
    # in case you really don't want to define your submit handler inside the
    # render function
    class Form(Protocol):
        name: Input[str]
        age: Input[int]

    class Example(Modal):
        def render(self) -> View:
            class ConcreteForm(Builder):
                name = Input("Name", PlainTextInput())
                age = Input("Age", NumberInput(is_decimal_allowed=False))

            return View(
                title="Example", builder=ConcreteForm, on_submit=("Ok", self.on_submit)
            )

        def on_submit(self, f: Form) -> OnSubmit:
            return None
