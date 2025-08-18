class Settings:
    def __init__(self, model_path_or_name: str, db_path: str = "sqliterag.db"):
        self.model_path_or_name = model_path_or_name
        self.db_path = db_path

        self.embedding_dim = 384
        self.vector_type = "FLOAT32"

        self.model_config = "n_ctx=384"  # See: https://github.com/sqliteai/sqlite-ai/blob/e172b9c9b9d76435be635d1e02c1e88b3681cc6e/src/sqlite-ai.c#L51-L57

        self.chunk_size = 256  # Maximum tokens per chunk
        self.chunk_overlap = 32  # Token overlap between chunks

        self.quantize_scan = True  # Whether to quantize the vector for faster search

