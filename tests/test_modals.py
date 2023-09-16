import datetime
from unittest import mock
from unittest.mock import Mock

from melax.modals import Actions, Button, Errors, Input, NumberInput, Ok, DatePicker


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
    i = Input("Enter a number", NumberInput(is_decimal_allowed=False)).error_if(
        lambda x: x == 7, "7 is unlucky"
    )
    i._block_id = "block_id"

    p = i._parse({"block_id": {"NumberInput": {"value": "7"}}})
    assert p == Errors({"block_id": "7 is unlucky"})

    p2 = i._parse({"block_id": {"NumberInput": {"value": "123"}}})
    assert p2 == Ok(123)


def test_actions_blocks() -> None:
    yay = Mock()
    nay = Mock()
    dob = Mock()

    yay_or_nay = Actions(
        yay=Button("Yay", value="yay", style="primary").on_pressed(yay),
        nay=Button("Nay", value="nay", style="danger").on_pressed(nay),
        dob=DatePicker().on_picked(dob),
    )
    yay_or_nay._block_id = "yay_or_nay"

    yay_action = {
        "action_id": "yay",
        "block_id": "yay_or_nay",
        "text": {"type": "plain_text", "text": "Yay", "emoji": True},
        "value": "yay",
        "style": "primary",
        "type": "button",
        "action_ts": "1694859764.558458",
    }

    yay_or_nay._on_action("yay", yay_action)
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

    yay_or_nay._on_action("dob", date_picker_action)
    dob.assert_called_with(today)

    submit_payload = {
        "yay_or_nay": {"dob": {"type": "datepicker", "selected_date": None}}
    }

    p = yay_or_nay._parse(submit_payload)
    assert isinstance(p, Ok)

    assert p.value == {"yay": "yay", "nay": "nay", "dob": None}
