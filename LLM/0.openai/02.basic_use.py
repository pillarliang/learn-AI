from openai import OpenAI
import os
from pprint import pprint

client = OpenAI()

models = client.models.list()  # list models
model_list = [model.id for model in models.data]

client.models.retrieve("gpt-4-vision-preview")  # retrieve a model


# ## 1. completion - 生成英文文本
# data = client.completions.create(
#     model="gpt-3.5-turbo-instruct",
#     prompt="Say this is a test",
#     max_tokens=7,
#     temperature=0,
# )
# print(data.choices[0].text)


# ## 2. completion - generate python code
# data = client.completions.create(
#     model="gpt-3.5-turbo-instruct",
#     prompt="generate python code about quick sort",
#     max_tokens=1000,
#     temperature=0,
# )
# text = data.choices[0].text
# print(text)
# exec(text)


# ## 3. chat completion - chatbot
# messages = [
#     {
#         "role": "user",
#         "content": "Hello",
#     }
# ]

# data = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
# new_message = data.choices[0].message
# new_message_dict = {"role": new_message.role, "content": new_message.content}
# messages.append(new_message_dict)

# new_chat = {
#     "role": "user",
#     "content": "1. When is it today? 2. when is it tomorrow?",
# }
# messages.append(new_chat)

# data = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
# new_message = data.choices[0].message
# print(new_message.content)


## 4. chat completion - multi-identity chatbot
messages = [
    {"role": "system", "content": "You're the helpful sports specialist."},
    {"role": "user", "content": "where is the place hold the 2008 Olympic Games?"},
]

data = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
)

message = data.choices[0].message.content
messages.append({"role": "assistant", "content": message})
# next round
messages.append({"role": "user", "content": "Which country won the most gold medals?"})
data = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
)
message = data.choices[0].message.content
print(message)
