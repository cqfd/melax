import os
import re
import datetime
import json

from enum import Enum
from typing import Any, Sequence

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_bolt.context import BoltContext
from slack_sdk import WebClient

from .modals import (
    Actions,
    Modal,
    View,
    Block,
    Builder,
    Input,
    Select,
    Divider,
    Section,
    PlainText,
    nested,
    OnSubmit,
    Errors,
    Push,
    PlainTextInput,
    Button,
    DatePicker,
    NumberInput,
    _modals,
)


class IceCream(Enum):
    CHOCOLATE = "chocolate"
    VANILLA = "vanilla"
    STRAWBERRY = "strawberry"


class ExampleModal(Modal):
    """
    Janky grab-bag of some things you can do.
    """

    # The modal's state.
    click_count: int = 0
    name: str | None = None

    def render(self) -> View:
        # Declare a collection of blocks, in such a way that the on_submit
        # below can get nice type-safety. (Looks weird to declare a new class
        # inside the render method, but it works.)
        class Form(Builder):
            name = Input(
                "Name" + "!" * self.click_count,
                PlainTextInput().on_enter_pressed(self.on_name_changed),
            )

            dob = Input("Date of birth", DatePicker().on_picked(self.on_date_picked))

            fav_number = Input(
                "Favorite number", NumberInput(is_decimal_allowed=False)
            ).error_if(lambda x: x == 7, "7 is unlucky")

            fav_ice_cream = Input(
                "Favorite ice cream",
                Select(
                    options=self.fav_ice_cream_options,
                )
                .map(lambda x: IceCream(x))
                .on_selected(self.on_ice_cream_picked),
            ).error_if(lambda flavor: flavor == IceCream.VANILLA, "Ew, vanilla")

            clickable = Section(
                PlainText("I'm hopefully clickable"),
                accessory=Button("Click me!", value="42")
                .on_pressed(self.on_click_str)
                .map(lambda x: -int(x))
                .on_pressed(self.on_click_int),
            )

            _divider = Divider().map(lambda _: 123)

            wow = (
                Section(PlainText(f"Click count: {self.click_count}"))
                if self.click_count > 2
                else None
            )
            huh = Divider().map(lambda _: "huh") if self.click_count > 3 else None

            # nesting is one escape valve from static types: you can do
            # whatever dynamic things you want when determing what keys to
            # include, at the cost of mypy not knowing as much about the type
            # of `form.extra`.
            extra = nested(
                something_else=Input("Something else", PlainTextInput()),
                and_one_more=nested(
                    thing=Input("Password", PlainTextInput()).error_unless(
                        lambda pw: pw == "open sesame", "Wrong!"
                    ),
                ),
            )

            yay_or_nay = Actions(
                yay=Button("Yay", value="yay", style="primary").on_pressed(self.yay),
                nay=Button("Nay", value="nay", style="danger").on_pressed(self.nay),
                dob=DatePicker(),
            )

        # Here the magic is that mypy knows everything about `form`!
        # Bit weird-looking that on_submit is nested inside the render method,
        # but that allows us to refer to the `Form` type. There are ways to get
        # around doing this with typing.Protocols, but this is a bit simpler.
        def on_submit(form: Form) -> OnSubmit:
            print(f"form={vars(form)}")
            # Signal what we want to do next via the return value.
            return Push(NiceToMeetYouModal(name=form.name))

        # Finally, render returns a View that bundles everything together.
        return View(
            title="Example" if self.name is None else f"Hello {self.name}!",
            # The blocks specification is the Form *class*; the on_submit
            # handler will receive an instance. Looks weird but actually
            # works pretty well in terms of types.
            blocks=Form,
            on_submit=("Click me!", on_submit),
        )

    def yay(self, value: str) -> None:
        print(f"Yay! {value=}")

    def nay(self, value: str) -> None:
        print(f"Nay! {value=}")

    def on_name_changed(self, name: str) -> None:
        print(f"Name changed: {name=}")
        self.name = name

    def fav_ice_cream_options(self, query: str) -> Sequence[Select.Option]:
        print(f"{query=}")
        return [
            Select.Option(text=flavor.name, value=flavor.value) for flavor in IceCream
        ]

    def on_click_str(self, value: str) -> None:
        assert isinstance(value, str)
        print(f"Clicked {value=}")
        # the modal will be re-rendered after action callbacks
        self.click_count += 1

    def on_click_int(self, value: int) -> None:
        assert isinstance(value, int)
        print(f"Clicked {value=}")
        # the modal will be re-rendered after action callbacks
        self.click_count += 1

    def on_date_picked(self, d: datetime.date) -> None:
        print(f"Date picked: {d=}")

    def on_ice_cream_picked(self, flavor: IceCream) -> None:
        print(f"Ice cream picked: {flavor=}")


class NiceToMeetYouModal(Modal):
    name: str

    def render(self) -> View:
        class Form(Builder):
            sound_good = (
                Input(
                    "Sound good?",
                    PlainTextInput().on_character_entered(self.on_character_entered),
                )
                .error_unless(lambda msg: len(msg) >= 5, "Gotta be at least 5 chars")
                .error_if(lambda msg: len(msg) > 10, "Too long")
            )
            dob = Section("Date of birth", accessory=DatePicker())
            button = Section(
                "Click me", accessory=Button("Click", value="ok").on_pressed(print)
            )

        def on_submit(form: Form) -> None:
            print(f"{form.sound_good=} {form.dob=} {form.button=}")

        return View(title="Nice to meet you!", blocks=Form, on_submit=("Ok", on_submit))

    def on_enter_pressed(self, msg_so_far: str) -> None:
        print(f"on_enter_pressed: {msg_so_far=}")

    def on_character_entered(self, msg_so_far: str) -> None:
        print(f"on_character_entered: {msg_so_far=}")


class TestModal(Modal):
    def render(self) -> View:
        class Form(Builder):
            foo = Section("Hi!", accessory=DatePicker())
            bar = Input("Bar", DatePicker())
            fav_color = Input(
                "Fav color", Select(options=[Select.Option(text="Blue", value="blue")])
            )

        return View(
            title="Test",
            blocks=Form,
            on_submit=("Ok", lambda form: print(form.fav_color)),
        )


app = App(token=os.environ.get("SLACK_BOT_TOKEN"))


@app.command("/do")  # <- or whatever your command is called
def handle_do(context: BoltContext, client: WebClient, body: dict[str, Any]) -> None:
    context.ack()

    # modal = NiceToMeetYouModal(name="Alan")
    modal = TestModal()

    client.views_open(trigger_id=body["trigger_id"], view=modal._to_slack_view_json())


@app.view("__melax__")
def handle_modal_submission(context: BoltContext, body: dict[str, Any]) -> None:
    context.ack()

    print(json.dumps(body, indent=2))

    private_metadata = json.loads(body["view"]["private_metadata"])
    modal_type = _modals[private_metadata["type"]]
    modal = modal_type.model_validate_json(private_metadata["value"])

    # *re*-render the modal, which had better produce the same view as
    # whatever thing on the user's screen led to this callback
    view = modal.render()
    blocks = view.blocks
    if isinstance(blocks, Block):
        if blocks._block_id is None:
            blocks._block_id = body["view"]["blocks"][0]["block_id"]

    # Parse whatever the user filled in
    result = view.blocks._parse(body["view"]["state"]["values"])
    assert result is not None
    if isinstance(result, Errors):
        context.ack(response_action="errors", errors=result.errors)
        return

    # Run the modal's on_submit handler, which returns whatever
    # we should do next.
    next_step = view.on_submit[1](result.value)
    match next_step:
        case None:
            context.ack(response_action="clear")
        case Push(next_modal):
            context.ack(response_action="push", view=next_modal._to_slack_view_json())
        case Errors(errors):
            context.ack(response_action="errors", errors=errors)
        case next_modal:
            context.ack(response_action="update", view=next_modal._to_slack_view_json())


@app.action(re.compile(r".*"))
def handle_block_actions(
    context: BoltContext, body: dict[str, Any], client: WebClient
) -> None:
    context.ack()

    print(json.dumps(body, indent=2))

    actions = body["actions"]
    assert len(actions) == 1, f"Weird, got more than one action: {actions}"
    action = actions[0]

    private_metadata = json.loads(body["view"]["private_metadata"])
    modal_type = _modals[private_metadata["type"]]
    modal = modal_type.model_validate_json(private_metadata["value"])

    # *re*-render the modal, which had better produce the same view as
    # whatever thing on the user's screen led to this callback
    view = modal.render()
    # Run the action handler
    view.blocks._on_block_action(action["block_id"], action["action_id"], action)
    # And then *re*-render the modal again, since its state may have changed
    client.views_update(
        view_id=body["view"]["id"],
        hash=body["view"]["hash"],
        view=modal._to_slack_view_json(),
    )


@app.options(re.compile(r".*"))
def handle_options(
    context: BoltContext, body: dict[str, Any], client: WebClient
) -> None:
    context.ack()
    print(json.dumps(body, indent=2))

    query = body["value"]

    private_metadata = json.loads(body["view"]["private_metadata"])
    modal_type = _modals[private_metadata["type"]]
    original_state = private_metadata["value"]
    modal = modal_type.model_validate_json(private_metadata["value"])

    view = modal.render()
    options = view.blocks._on_block_options(body["block_id"], body["action_id"], query)
    new_state = modal.model_dump_json()
    assert (
        new_state == original_state
    ), f"Option callbacks can't modify the state of the modal! {new_state=} != {original_state=}"

    context.ack(options=[o.to_slack_json() for o in options])


if __name__ == "__main__":
    SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN")).start()  # type: ignore
