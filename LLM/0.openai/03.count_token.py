import tiktoken


# # encoding_name = tiktoken.encoding_for_model("gpt-3.5-turbo")  # 获取一个模型的编码方式
# # encoding = tiktoken.get_encoding("cl100k_base")  # 第一次运行需联网下载。后续不需要。
# encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # 可以自动加载给定模型名称的正确编码。


# ##### The .encode() method converts a text string into a list of token integers. #####
# encoding.encode("tiktoken is great!")  # [83, 1609, 5963, 374, 2294, 0]


# def num_tokens_from_string(string: str, encoding_name: str) -> int:
#     """Return the number of tokens in a string."""
#     encoding = tiktoken.get_encoding(encoding_name)
#     return len(encoding.encode(string))


# print(num_tokens_from_string("tiktoken is great!", "cl100k_base"))


# ##### .decode()将一个token整数列表转换为字符串。
# encoding.decode([83, 1609, 5963, 374, 2294, 0])  # tiktoken is great! #####

# ## .decode_single_token_bytes() 安全地将单个整数token转换为其表示的字节。
# print(
#     [
#         encoding.decode_single_token_bytes(token)
#         for token in [83, 1609, 5963, 374, 2294, 0]
#     ]
# )  # [b't', b'ik', b'token', b' is', b' great', b'!'] 在字符串前面的b表示这些字符串是字节字符串。


# ##### Comparing encodings #####
# # 不同的编码方式在分割单词、处理空格和非英文字符方面存在差异。
# def compare_encodings(example: str) -> None:
#     """Compare the encodings of a string in different encodings."""
#     print(f"\nExample: {example}")

#     for encoding_name in ["cl100k_base", "gpt2", "p50k_base"]:
#         encoding = tiktoken.get_encoding(encoding_name)
#         token_integers = encoding.encode(example)
#         num_tokens = len(token_integers)
#         token_bytes = [
#             encoding.decode_single_token_bytes(token) for token in token_integers
#         ]

#         print(f"\n{encoding_name}: {num_tokens} tokens")
#         print(f"token integers:{token_integers}")
#         print(f"token bytes: {token_bytes}")


# print(compare_encodings("tiktoken is great!"))
# print(compare_encodings("2 + 2 = 4"))
# print(compare_encodings("お誕生日おめでとう"))


##### Counting tokens for chat completions API calls #####
# 计算传递给gpt-3.5-turbo或gpt-4的消息中的token数量。
# 请注意，从消息中计算token的确切方式可能因模型而异。请将下面函数中的计数视为估计值，并非永恒保证。
# 特别地，在使用可选功能输入(input)的请求上方会消耗额外的token。
def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """Return the number of tokens used by a list of messages."""
    # 尝试获取模型的编码
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # 如果模型没有找到，使用 cl100k_base 编码并给出警告
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    # 针对不同的模型设置token数量
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # 每条消息遵循 {role/name}\n{content}\n 格式
        tokens_per_name = -1  # 如果有名字，角色会被省略
    elif "gpt-3.5-turbo" in model:
        # 对于 gpt-3.5-turbo 模型可能会有更新，此处返回假设为 gpt-3.5-turbo-0613 的token数量，并给出警告
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        # 对于 gpt-4 模型可能会有更新，此处返回假设为 gpt-4-0613 的token数量，并给出警告
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    elif model in {"davinci", "curie", "babbage", "ada"}:
        print(
            "Warning: gpt-3 related model is used. Returning num tokens assuming gpt2."
        )
        encoding = tiktoken.get_encoding("gpt2")
        num_tokens = 0
        # only calc the content
        for message in messages:
            for key, value in message.items():
                if key == "content":
                    num_tokens += len(encoding.encode(value))
        return num_tokens
    else:
        # 对于没有实现的模型，抛出未实现错误
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    # 计算每条消息的token数
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # 每条回复都以助手为首
    return num_tokens


from openai import OpenAI

client = OpenAI()


example_messages = [
    {
        "role": "system",
        "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
    },
    {
        "role": "system",
        "name": "example_user",
        "content": "New synergies will help drive top-line growth.",
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": "Things working well together will increase revenue.",
    },
    {
        "role": "system",
        "name": "example_user",
        "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": "Let's talk later when we're less busy about how to do better.",
    },
    {
        "role": "user",
        "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",
    },
]

for model in [
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo",
    "gpt-4-0613",
    "gpt-4",
]:
    print(model)
    # example token count from the function defined above
    print(
        f"{num_tokens_from_messages(example_messages, model)} prompt tokens counted by num_tokens_from_messages()."
    )
    # example token count from the OpenAI API
    # OpenAI Python SDK v1.0 更新后的使用方式
    completion = client.chat.completions.create(
        model=model,
        messages=example_messages,
        temperature=0,
        max_tokens=1,  # we're only counting input tokens here, so let's not waste tokens on the output
    )
    print(f"{completion.usage.prompt_tokens} prompt tokens counted by the OpenAI API.")
    print()
