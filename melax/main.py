import os
import re
import datetime
import json

from enum import Enum
from typing import Any

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_bolt.context import BoltContext
from slack_sdk import WebClient

from .modals import (
    Modal,
    View,
    Blocks,
    Input,
    Select,
    Divider,
    Section,
    PlainText,
    nested,
    OnSubmit,
    Errors,
    Push,
    Option,
    PlainTextInput,
    Button,
    DatePicker,
    NumberInput,
    sequence,
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

    def render(self) -> View:
        # Declare a collection of blocks, in such a way that the on_submit
        # below can get nice type-safety. (Looks weird to declare a new class
        # inside the render method, but it works.)
        class Form(Blocks):
            name = Input("Name" + "!" * self.click_count, PlainTextInput())

            dob = Input("Date of birth", DatePicker(on_selection=self.on_date_picked))

            fav_number = Input("Favorite number", NumberInput(is_decimal_allowed=False))

            fav_things = sequence(
                ("first", Input("Favorite thing 1", PlainTextInput())),
                ("second", Input("Favorite thing 2", PlainTextInput())),
            )

            fav_ice_cream = Input(
                "Favorite ice cream",
                Select(
                    options=self.fav_ice_cream_options,
                    on_selection=self.on_ice_cream_picked,
                ).map(lambda x: IceCream(x))
            )

            clickable = Section(
                PlainText("I'm hopefully clickable"),
                accessory=Button("Click me!", value="42", on_click=self.on_click),
            )

            _divider = Divider().map(lambda _: 123)

            # nesting is one escape valve from static types: you can do
            # whatever dynamic things you want when determing what keys to
            # include, at the cost of mypy not knowing as much about the type
            # of `form.extra`.
            extra = nested(
                something_else=Input("Something else", PlainTextInput()),
                and_one_more=nested(thing=Input("Thing", PlainTextInput())),
            )

        # render of course has access to the modal's state
        if self.click_count > 2:
            # Eh, experimenting with escape hatches from type-safety, when you
            # want to do something super dynamic. Don't really like this.
            Form.add("wow", Section(PlainText("Wow, you really click a lot!")))

        # Here the magic is that mypy knows everything about `form`!
        # Bit weird-looking that on_submit is nested inside the render method,
        # but that allows us to refer to the `Form` type. There are ways to get
        # around doing this with typing.Protocols, but this is a bit simpler.
        def on_submit(form: Form) -> OnSubmit:
            print(f"form={vars(form)}")

            # Experimenting with ways to more-or-less ergonomically express errors.
            # This is made a little tricky because these modals support "nested"
            # blocks, which means it's slightly trickier to understand what their
            # block_ids are.
            errors = []
            if form.fav_number == 7:
                errors.append(Form.fav_number.error("7 is a bad number"))
            if "wow" in form and form.fav_ice_cream is IceCream.VANILLA:
                errors.append(Form.fav_ice_cream.error("Ew, vanilla"))
            and_one_more = form.extra["and_one_more"]
            assert isinstance(and_one_more, dict)
            if and_one_more["thing"] != "open sesame":
                errors.append(Form.extra["and_one_more"]["thing"].error("Guess again"))
            if form.fav_things[0] == form.fav_things[1]:
                errors.extend(
                    [
                        Form.fav_things[0].error("Can't be the same"),
                        Form.fav_things[1].error("Can't be the same"),
                    ]
                )
            print(f"{errors=}")
            if errors:
                return Errors(*errors)

            # reveal_type(form.name) -> str, etc
            print(f"{form.name=} {form.dob=} {form.fav_number=} {self.click_count=}")

            # Signal what we want to do next via the return value.
            return Push(NiceToMeetYouModal(name=form.name))

        # Finally, render returns a View that bundles everything together.
        return View(
            title="Example",
            # The blocks specification is the Form *class*; the on_submit
            # handler will receive an instance. Looks weird but actually
            # works pretty well in terms of types.
            blocks=Form,
            on_submit=("Click me!", on_submit),
        )

    def fav_ice_cream_options(self, query: str) -> list[Option]:
        print(f"{query=}")
        return [Option(text=flavor.name, value=flavor.value) for flavor in IceCream]

    def on_click(self, value: str) -> None:
        print(f"Clicked {value=}")
        # the modal will be re-rendered after action callbacks
        self.click_count += 1

    def on_date_picked(self, d: datetime.date) -> None:
        print(f"Date picked: {d=}")

    def on_ice_cream_picked(self, flavor: str) -> None:
        print(f"Ice cream picked: {flavor=}")


class NiceToMeetYouModal(Modal):
    name: str

    def render(self) -> View:
        class Form(Blocks):
            msg = Section(PlainText(f"Nice to meet you {self.name}!"))

        return View(
            title="Nice to meet you!",
            blocks=Form,
            on_submit=("Ok", lambda _: print("All done!")),
        )


app = App(token=os.environ.get("SLACK_BOT_TOKEN"))


@app.command("/do")  # <- or whatever your command is called
def handle_do(context: BoltContext, client: WebClient, body: dict[str, Any]) -> None:
    context.ack()

    m = ExampleModal()

    j = m.to_slack_view_json()
    print(json.dumps(j, indent=2))

    client.views_open(
        trigger_id=body["trigger_id"],
        view=j,
    )


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
    # Parse whatever the user filled in
    result = view.blocks._parse(body["view"]["state"]["values"])
    # Run the modal's on_submit handler, which returns whatever
    # we should do next.
    next_step = view.on_submit[1](result)

    match next_step:
        case None:
            context.ack(response_action="clear")
        case Push(next_modal):
            context.ack(response_action="push", view=next_modal.to_slack_view_json())
        case Errors(errors):
            context.ack(response_action="errors", errors=errors)
        case next_modal:
            context.ack(response_action="update", view=next_modal.to_slack_view_json())


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
        view=modal.to_slack_view_json(),
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
