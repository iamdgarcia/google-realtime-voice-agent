import re

_SENTENCE_END = re.compile(r'[.!?](?=\s|$)')


class SentenceChunker:
    def __init__(self) -> None:
        self._buf = ""

    def push(self, token: str) -> list[str]:
        self._buf += token
        sentences: list[str] = []
        while True:
            match = _SENTENCE_END.search(self._buf)
            if not match:
                break
            end = match.end()
            sentences.append(self._buf[:end].strip())
            self._buf = self._buf[end:].lstrip()
        return sentences

    def flush(self) -> str:
        remainder = self._buf.strip()
        self._buf = ""
        return remainder

    def reset(self) -> None:
        self._buf = ""
