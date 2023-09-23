import datetime
import json
import os
import re
from enum import Enum
from typing import Any, Sequence

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_bolt.context import BoltContext
from slack_sdk import WebClient

from .blocks import (
    Actions,
    Blocks,
    Builder,
    Divider,
    Input,
    Section,
    blocks,
)
from .elements import (
    Button,
    DatePicker,
    NumberInput,
    PlainTextInput,
    Select,
    UsersSelect,
)
from .messages import Message, _messages
from .modals import Modal, OnSubmit, View, _modals
from .types import Bind, Errors, Ok, PlainText


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
            extra = blocks(
                dict(
                    something_else=Input("Something else", PlainTextInput()),
                    and_one_more=blocks(
                        dict(
                            thing=Input("Password", PlainTextInput()).error_if(
                                lambda pw: pw != "open sesame", "Wrong!"
                            )
                        ),
                    ),
                )
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
            return Modal.Push(NiceToMeetYouModal(name=form.name))

        # Finally, render returns a View that bundles everything together.
        return View(
            title="Example" if self.name is None else f"Hello {self.name}!",
            # The blocks specification is the Form *class*; the on_submit
            # handler will receive an instance. Looks weird but actually
            # works pretty well in terms of types.
            builder=Form,
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
                .error_if(lambda msg: len(msg) < 5, "Gotta be at least 5 chars")
                .error_if(lambda msg: 10 < len(msg), "Too long")
            )
            dob = Section("Date of birth", accessory=DatePicker())
            button = Section(
                "Click me", accessory=Button("Click", value="ok").on_pressed(print)
            )

        def on_submit(form: Form) -> None:
            print(f"{form.sound_good=} {form.dob=} {form.button=}")

        return View(
            title=f"Nice to meet you {self.name}!",
            builder=Form,
            on_submit=("Ok", on_submit),
        )

    def on_enter_pressed(self, msg_so_far: str) -> None:
        print(f"on_enter_pressed: {msg_so_far=}")

    def on_character_entered(self, msg_so_far: str) -> None:
        print(f"on_character_entered: {msg_so_far=}")


class TalkModal(Modal):
    name: str | None = None

    num_friends_at_work: int | None = None

    def render(self) -> View:
        class Form(Builder):
            name = Input(
                "What is your name?",
                PlainTextInput().on_character_entered(self.name_changed),
            )
            _greeting = (
                Section(f"Wow, {self.name} is a nice name 😌")
                if self.name is not None
                else None
            )

            age = Input("What is your age?", NumberInput(is_decimal_allowed=True))
            dob = Input("Date of birth", DatePicker())

            num_friends_at_work = (
                Input(
                    "How many friends do you have at work?",
                    NumberInput(is_decimal_allowed=False),
                )
                if self.num_friends_at_work is None
                else None
            )

            friends_at_work = blocks(
                {
                    str(i): Input(f"Friend {i}", UsersSelect())
                    for i in range(self.num_friends_at_work or 0)
                }
            )

        def on_submit(form: Form) -> OnSubmit:
            if self.num_friends_at_work is None:
                assert form.num_friends_at_work is not None
                self.num_friends_at_work = form.num_friends_at_work
                return self

            print(f"{form.name=} {form.age=} {form.dob=}")
            for friend in form.friends_at_work.values():
                print(f"{friend} is one of your friends at work!")
            return None

        return View(
            title="I'm the title" if self.name is None else f"Hi {self.name}!",
            builder=Form,
            on_submit=("Ok", on_submit),
        )

    def name_changed(self, name: str) -> None:
        print(f"Name changed: {name=}")
        self.name = name


class TestModal(Modal):
    name: str
    age: int
    keys: list[str] = ["A", "B"]

    def render(self) -> View:
        def on_best_friend_selected(user_id: str) -> None:
            print(f"Best friend selected: {user_id}")
            print(f"{locals()=}")

        class Form(Builder):
            foo = Section("Hi!", accessory=DatePicker())
            bar = Input("Bar", DatePicker())
            fav_color = Input(
                "Fav color", Select(options=[Select.Option(text="Blue", value="blue")])
            )
            best_friend = Input(
                "Best friend", UsersSelect().on_selected(on_best_friend_selected)
            )
            actions = Actions(
                button=Button("Click", value="ok").on_pressed(print),
            )

            stuff = blocks({k: Input(k, PlainTextInput()) for k in self.keys})
            reorder = Actions(
                button=Button("Reorder").on_pressed(lambda _: self.keys.reverse()),
            )

            not_actually_optional = Input(
                "Not actually optional",
                PlainTextInput(initial_value="hi"),
                optional=True,
            ).map_or_error_msg(
                lambda x: Ok(x) if x is not None else "Actually this is required lol"
            )

        def on_submit(form: Form) -> None:
            print(f"Cool! {vars(form)=}")

        return View(title="Test", builder=Form, on_submit=("Ok", on_submit))


class CoolnessCheck(Message):
    click_count: int = 0

    @property
    def text(self) -> str:
        return "Do you think this is cool?"

    def render(self) -> Blocks[Any]:
        if self.click_count > 0:
            return Section("Confirmed cool :thumbsup:")

        return Section(
            "Do you think this is cool?",
            accessory=Button("Yes", value="yes").on_pressed(self.on_button_pressed),
        )

    def on_button_pressed(self, v: str) -> None:
        print(f"CoolnessCheck {v=}")
        self.click_count += 1


class BindingCheck(Modal):
    name_element: Bind[str] = Bind()

    def render(self) -> View:
        class Form(Builder):
            name = Input("Name", PlainTextInput().bind(self.name_element)).error_if(
                lambda name: len(name) < 5, "Too short"
            )
            dob = Input("Age", DatePicker().on_picked(self.on_dob_picked))

        return View(
            title="Binding check", builder=Form, on_submit=("Ok", lambda _: None)
        )

    def on_dob_picked(self, d: datetime.date) -> None:
        print(f"BindingCheck: on_dob_picked {d=} {self.name_element.value=}")


class EdgeCase(Modal):
    fav_ice_cream: Bind[str] = Bind()

    def render(self) -> View:
        class Form(Builder):
            fav_ice_cream = Input(
                "Fav ice cream:", Select(options=["chocolate", "vanilla", "strawberry"])
            ).bind(self.fav_ice_cream)
            button = Section(
                "Click me",
                accessory=Button("Click me", value="ok").on_pressed(self.on_pressed),
            )

        def on_submit(f: Form) -> None:
            print(f"EdgeCase: on_submit {vars(f)}")
            return None

        return View(
            title="Edge case",
            builder=Form,
            on_submit=("Ok", on_submit),
        )

    def on_pressed(self, v: str) -> None:
        print(f"Button pressed: {v=} {self.fav_ice_cream.value=}")


app = App(token=os.environ.get("SLACK_BOT_TOKEN"))


def inflate_block_ids(values: dict[str, Any]) -> dict[str, Any]:
    def set(result: dict[str, Any], path: list[str], value: Any) -> None:
        if len(path) == 1:
            result[path[0]] = value
            return
        prefix = path[0]
        rest = path[1:]
        if prefix not in result:
            result[prefix] = {}
        set(result[prefix], rest, value)

    result: dict[str, Any] = {}
    for k, v in values.items():
        set(result, json.loads(k), v)

    return result


@app.command("/do")  # <- or whatever your command is called
def handle_do(context: BoltContext, client: WebClient, body: dict[str, Any]) -> None:
    context.ack()

    print(json.dumps(body, indent=2))

    # modal = NiceToMeetYouModal(name="Alan")
    # modal = TestModal(name="Alan", age=38)
    modal_type, *args = body["text"].split()
    kwargs = dict(a.split("=", 1) for a in args)
    modal = next(modal for name, modal in _modals.items() if modal_type in name)(
        **kwargs
    )

    print(json.dumps(modal._to_slack_view_json(), indent=2))

    client.views_open(trigger_id=body["trigger_id"], view=modal._to_slack_view_json())


@app.view("__melax__")
def handle_modal_submission(context: BoltContext, body: dict[str, Any]) -> None:
    context.ack()

    print(json.dumps(body, indent=2))

    values = inflate_block_ids(body["view"]["state"]["values"])

    private_metadata = json.loads(body["view"]["private_metadata"])
    modal_type = _modals[private_metadata["type"]]
    modal = modal_type.model_validate(private_metadata["value"])

    client = context.client
    assert client is not None
    modal._client = client

    # *re*-render the modal, which had better produce the same view as
    # whatever thing on the user's screen led to this callback
    view = modal.render()
    # Parse whatever the user filled in
    result = view.builder._parse(values)
    assert result is not None
    if isinstance(result, Errors):
        context.ack(response_action="errors", errors=result._to_slack_json())
        return

    # Run the modal's on_submit handler, which returns whatever
    # we should do next.
    next_step = view.on_submit[1](result.value)
    match next_step:
        case None:
            context.ack(response_action="clear")
        case Modal.Push(next_modal):
            context.ack(response_action="push", view=next_modal._to_slack_view_json())
        case Errors(_):
            context.ack(response_action="errors", errors=next_step._to_slack_json())
        case next_modal:
            context.ack(response_action="update", view=next_modal._to_slack_view_json())


@app.action(re.compile(r".*"))
def handle_block_actions(
    context: BoltContext, body: dict[str, Any], client: WebClient
) -> None:
    context.ack()

    print(json.dumps(body, indent=2))

    values = inflate_block_ids(body["view"]["state"]["values"])

    actions = body["actions"]
    assert len(actions) == 1, f"Weird, got more than one action: {actions}"
    action = actions[0]

    if "message" in body:
        private_metadata = json.loads(
            body["message"]["blocks"][-1]["elements"][0]["image_url"]
        )
        message_type = _messages[private_metadata["type"]]
        message = message_type.model_validate(private_metadata["value"])
        message.render()._on_block_action(
            json.loads(action["block_id"]), action["action_id"], action
        )

        client.chat_update(
            channel=body["channel"]["id"],
            text=message.text,
            ts=body["message"]["ts"],
            blocks=message._to_slack_blocks_json(),  # type: ignore
        )

        return

    private_metadata = json.loads(body["view"]["private_metadata"])
    modal_type = _modals[private_metadata["type"]]
    modal = modal_type.model_validate(private_metadata["value"])

    # *re*-render the modal, which had better produce the same view as
    # whatever thing on the user's screen led to this callback
    view = modal.render()
    # Hack: parse so we can wire up any bindings *before* running the
    # actual action handler.
    view.builder._parse(values)
    # Run the action handler
    view.builder._on_block_action(
        json.loads(action["block_id"]), action["action_id"], action
    )
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
    modal = modal_type.model_validate(private_metadata["value"])

    view = modal.render()
    options = view.builder._on_block_options(
        json.loads(body["block_id"]), body["action_id"], query
    )
    new_state = modal.model_dump()
    assert (
        new_state == original_state
    ), f"Option callbacks can't modify the state of the modal! {new_state=} != {original_state=}"

    context.ack(options=[o.to_slack_json() for o in options])


if __name__ == "__main__":
    SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN")).start()  # type: ignore
