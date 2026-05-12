from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings as _default_settings


def get_chunker(cfg=None, embedder=None):
    """Return a text splitter based on settings.chunk_strategy."""
    cfg = cfg or _default_settings
    if cfg.chunk_strategy == "semantic":
        from langchain_experimental.text_splitter import SemanticChunker
        if embedder is None:
            raise ValueError(
                "An embedder must be provided when chunk_strategy='semantic'."
            )
        return SemanticChunker(embedder)
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
    )
