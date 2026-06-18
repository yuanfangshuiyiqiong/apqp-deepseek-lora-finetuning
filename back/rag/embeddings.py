import hashlib
import math
import re
from typing import List


class HashingEmbeddingModel:
    """Deterministic local embedding fallback with no model download required."""

    def __init__(self, dimension: int = 384):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    def embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        for token in self._tokens(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _tokens(self, text: str) -> List[str]:
        raw_tokens = re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)
        tokens: List[str] = []
        for token in raw_tokens:
            tokens.append(token)
            cjk_chars = [char for char in token if "\u4e00" <= char <= "\u9fff"]
            if len(cjk_chars) > 1:
                tokens.extend("".join(cjk_chars[index : index + 2]) for index in range(len(cjk_chars) - 1))
        return tokens
