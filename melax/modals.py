import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, Union

import pydantic
import slack_sdk

from .blocks import Builder, Errors
from .messages import Message
from .types import PlainText

_modals: dict[str, type["Modal"]] = {}

SomeBuilderSubclass = TypeVar("SomeBuilderSubclass", bound=Builder)


class Modal(ABC, pydantic.BaseModel):
    """
    What you'll subclass in order to define a modal.

    Modals are *stateful* and must be serializable. I'm subclassing
    pydantic.BaseModel as a hacky way to get serializability; make sure
    pydantic knows how to serialize whatever state variables you use.
    """

    @abstractmethod
    def render(self) -> "View":
        ...

    _client: slack_sdk.WebClient = pydantic.PrivateAttr()

    def dm(self, user_id: str, msg: "str | Message") -> None:
        if isinstance(msg, str):
            self._client.chat_postMessage(
                channel=user_id,
                text=msg,
            )
        else:
            self._client.chat_postMessage(
                channel=user_id,
                text=msg.text,
                blocks=msg._to_slack_blocks_json(),  # type: ignore
            )

    def __init_subclass__(cls) -> None:
        fully_qualified_class_name = f"{cls.__module__}.{cls.__name__}"
        assert (
            fully_qualified_class_name not in _modals
        ), f"{fully_qualified_class_name} has already been registered as a Slack modal type!"
        _modals[fully_qualified_class_name] = cls

    def _to_slack_view_json(self) -> dict[str, Any]:
        view = self.render()
        fully_qualified_class_name = (
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        return {
            "type": "modal",
            "title": PlainText(view.title)._to_slack_json(),
            "blocks": view.blocks._to_slack_blocks_json(),
            "submit": {"type": "plain_text", "text": view.on_submit[0]},
            "callback_id": "__melax__",
            "private_metadata": json.dumps(  # <- needs to be a string
                {"type": fully_qualified_class_name, "value": self.model_dump()}
            ),
        }

    class View:
        """
        Bundles a collection of blocks with a submit handler, in a type-safe way.

        https://api.slack.com/reference/surfaces/views
        """

        def __init__(
            self,
            title: str,
            blocks: type[SomeBuilderSubclass],
            on_submit: tuple[(str, Callable[[SomeBuilderSubclass], "OnSubmit"])],
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


OnSubmit = Union[
    Modal,  # continue being a (possibly different) modal
    Errors,  # validation errors
    Modal.Push,  # push a new modal
    None,  # finish, clearing the whole modal stack.
]
"""
Your options for what to do after the user submits a modal.
"""
