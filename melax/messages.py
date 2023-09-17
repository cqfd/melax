import json
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

import pydantic

from .blocks import Blocks, Context, Image
from .types import JSON

_messages: dict[str, type["Message"]] = {}


class Message(ABC, pydantic.BaseModel):
    @abstractmethod
    def render(self) -> Blocks[Any]:
        ...

    @property
    @abstractmethod
    def text(self) -> str:
        ...

    def _to_slack_blocks_json(self) -> Sequence[Mapping[str, JSON]]:
        blocks = self.render()
        result = [b for b in blocks._to_slack_blocks_json()]
        private_metadata = json.dumps(
            {
                "type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                "value": self.model_dump(),
            }
        )
        if blocks.is_interactive():
            result.append(
                Context(
                    Image(
                        url=private_metadata,
                        alt_text="ignore this implementation detail 😉",
                    )
                )._to_slack_json(),
            )
        return result

    def __init_subclass__(cls) -> None:
        fully_qualified_class_name = f"{cls.__module__}.{cls.__name__}"
        assert (
            fully_qualified_class_name not in _messages
        ), f"{fully_qualified_class_name} has already been registered as a Slack message type!"
        _messages[fully_qualified_class_name] = cls
