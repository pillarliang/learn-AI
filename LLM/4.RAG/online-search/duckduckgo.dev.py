import re
from duckduckgo_search import DDGS
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer


# TODO: differences between sync and async


def get_ddg_search_result(query: str) -> list[str]:
    result = get_ddg_search(query)

    urls = []
    for res in result:
        url = res["href"]
        urls.append(url)

    pages = get_webpage_from_urls(urls)

    content = []
    for page in pages:
        page_text = re.sub("\n\n+", "\n", page.page_content)
        text = _truncate(page_text)
        content.append(text)

    return content


def get_ddg_search(query: str, max_result: int = 5) -> list:
    results = DDGS().text(keywords=query, max_results=max_result, region="cn-zh")
    return results


def get_webpage_from_urls(urls: list[str]):
    loader = AsyncChromiumLoader(urls)
    html = loader.load()

    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        html, tags_to_extract=["p"], remove_unwanted_tags=["a"]
    )

    return docs_transformed


def _truncate(text):
    words = text.split()
    truncated = " ".join(words[:400])

    return truncated


if __name__ == "__main__":
    # print(get_ddg_search("ios 18 新特性"))
    print(get_ddg_search_result("ios 18 新特性"))
