"""Token counting shared by chunking and the /query source budget."""

from llama_index.core.utils import get_tokenizer


def count_tokens(text: str) -> int:
    """Token count with LlamaIndex's default tokenizer (tiktoken cl100k_base, bundled offline).

    SentenceSplitter measures chunk_size with the same tokenizer. Claude's tokenizer differs, so
    this is an estimate of what the LLM bills.
    """
    return len(get_tokenizer()(text))
