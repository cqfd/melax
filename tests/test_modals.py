from unittest import mock
from unittest.mock import Mock

from melax.modals import Button, Errors, Input, NumberInput, Ok


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
