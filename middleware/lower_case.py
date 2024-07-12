from typing import List

from rasa.engine.graph import GraphComponent
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.engine.recipes.default_recipe import DefaultV1Recipe


@DefaultV1Recipe.register(
    component_types=[DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER],
    is_trainable=False,
)
class TextCorrectionMiddleware(GraphComponent):

    def __init__(self, component_config=None):
        super().__init__(component_config)

    def process(self, messages: List[Message], **kwargs) -> List[Message]:
        for message in messages:
            text = message.get(TEXT)
            if text:
                print(f"Original Text: {text}")  # Print original text
                message.set(TEXT, text.lower())
                print(f"Lowercased Text: {message.get(TEXT)}")
        return messages
