import json
from openai import OpenAI
from pprint import pprint
import os
import requests
from termcolor import colored
from tenacity import retry, wait_random_exponential, stop_after_attempt

GPT_MODEL = "gpt-3.5-turbo"
client = OpenAI()


# # 使用了retry库，指定在请求失败时的重试策略。
# # 这里设定的是指数等待（wait_random_exponential），时间间隔的最大值为40秒，最多重试3次（stop_after_attempt(3)）。
# @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
# def chat_completion_request(
#     messages, functions=None, function_call=None, model=GPT_MODEL
# ):
#     return client.chat.completions.create(
#         model=model,
#         messages=messages,
#         tools=functions,
#     )


def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "tool": "magenta",
    }

    # 遍历消息列表
    for message in messages:
        # 如果消息的角色是"system"，则用红色打印“content”
        if message["role"] == "system":
            print(
                colored(
                    f"system: {message['content']}\n", role_to_color[message["role"]]
                )
            )

        # 如果消息的角色是"user"，则用绿色打印“content”
        elif message["role"] == "user":
            print(
                colored(f"user: {message['content']}\n", role_to_color[message["role"]])
            )

        elif message["role"] == "assistant":
            print(
                colored(
                    f"assistant: {message['content']}\n", role_to_color[message["role"]]
                )
            )

        elif message["role"] == "tool":
            print(
                colored(f"tool: {message['content']}\n", role_to_color[message["role"]])
            )


# 第一个字典定义了一个名为"get_current_weather"的功能
functions = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",  # 功能的名称
            "description": "Get the current weather",  # 功能的描述
            "parameters": {  # 定义该功能需要的参数
                "type": "object",
                "properties": {  # 参数的属性
                    "location": {  # 地点参数
                        "type": "string",  # 参数类型为字符串
                        "description": "The city and state, e.g. San Francisco, CA",  # 参数的描述
                    },
                    "format": {  # 温度单位参数
                        "type": "string",  # 参数类型为字符串
                        "enum": ["celsius", "fahrenheit"],  # 参数的取值范围
                        "description": "The temperature unit to use. Infer this from the users location.",  # 参数的描述
                    },
                },
                "required": ["location", "format"],  # 该功能需要的必要参数
            },
        },
    },
]

messages = []

# add system message
messages.append(
    {
        "role": "system",
        "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
    }
)

# add user message
messages.append(
    {
        "role": "user",
        "content": "What's the weather like today?",
    }
)

chat_response = client.chat.completions.create(
    model=GPT_MODEL,
    messages=messages,
    tools=functions,
)

content = chat_response.choices[0].message.content
messages.append({"role": "assistant", "content": content})

# 向messages列表添加一条用户角色的消息，用户告知他们在苏格兰的格拉斯哥
messages.append(
    {"role": "user", "content": "I'm in Shanghai, China."}  # 消息的角色是"user"  # 用户的消息内容
)

chat_response = client.chat.completions.create(
    model=GPT_MODEL,
    messages=messages,
    tools=functions,
    tool_choice="auto",
)

# get agruments
print(chat_response.choices[0].message.tool_calls[0].function.arguments)

# message = chat_response.choices[0].message.tool_calls[0].function
# messages.append({"role": "tool", "content": message})
# print(messages)
# pretty_print_conversation(messages)
