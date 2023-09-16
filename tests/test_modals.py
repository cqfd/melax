import datetime
from unittest import mock
from unittest.mock import Mock

from melax.modals import (
    Actions,
    Builder,
    Button,
    DatePicker,
    Errors,
    Input,
    NumberInput,
    Ok,
    Option,
    PlainTextInput,
    Section,
    Select,
)


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

    Form._on_block_action("name", "PlainTextInput", {"value": "Foo"})
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
    class Form(Builder):
        i = Input("Enter a number", NumberInput(is_decimal_allowed=False)).error_if(
            lambda x: x == 7, "7 is unlucky"
        )

    p = Form._parse({"i": {"NumberInput": {"value": "7"}}})
    assert p == Errors({"i": "7 is unlucky"})

    p2 = Form._parse({"i": {"NumberInput": {"value": "123"}}})
    assert isinstance(p2, Ok)
    assert p2.value.i == 123


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

    Form._on_block_action("yay_or_nay", "yay", yay_action)
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

    Form._on_block_action("yay_or_nay", "dob", date_picker_action)
    dob.assert_called_with(today)

    submit_payload = {
        "yay_or_nay": {"dob": {"type": "datepicker", "selected_date": None}}
    }

    p = Form._parse(submit_payload)
    assert isinstance(p, Ok)
    assert p.value.yay_or_nay == {"yay": "yay", "nay": "nay", "dob": None}


def test_select_elements() -> None:
    chocolate = Option("Chocolate", "chocolate")
    vanilla = Option("Vanilla", "vanilla")
    static_options = [chocolate, vanilla]

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
    assert p.value.fav_ice_cream

    def external_options(query: str) -> list[Option]:
        return [
            o for o in static_options if o.text.lower().startswith(query)
        ]
    mock = Mock(side_effect=external_options)

    class Dynamic(Builder):
        fav_ice_cream = Input("Favorite ice cream", Select(options=mock))

    opts = Dynamic._on_block_options("fav_ice_cream", "Select", "choco")
    mock.assert_called_with("choco")
    assert opts == [chocolate]
