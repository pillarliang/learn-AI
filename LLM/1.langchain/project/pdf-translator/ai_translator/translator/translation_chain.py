from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


class TranslationChain:
    def __init__(self, model_name: str = "gpt-3.5-turbo", verbose: bool = True):
        system_message = """You are a translation expert, proficient in various languages. \n
            Translates {source_language} to {target_language}."""

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_message
        )

        human_message = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_message)

        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        chat = ChatOpenAI(model_name=model_name, temperature=0, verbose=verbose)
        self.chain = LLMChain(llm=chat, prompt=chat_prompt_template, verbose=verbose)

    def run(self, text: str, source_language: str, target_language: str) -> (str, bool):
        result = ""
        try:
            result = self.chain.run(
                {
                    "text": text,
                    "source_language": source_language,
                    "target_language": target_language,
                }
            )
        except Exception as e:
            return result, False

        return result, True
