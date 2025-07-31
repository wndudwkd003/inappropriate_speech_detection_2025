import re

def chunk_texts(texts: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """긴 문단들을 chunk_size와 overlap 기준으로 분할 + 전처리"""
    chunks = []
    for paragraph in texts:
        paragraph = preprocess_text(paragraph)  # 전처리 추가
        words = paragraph.split()
        if len(words) <= chunk_size:
            chunks.append(paragraph)
            continue
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start += chunk_size - chunk_overlap
    return chunks




def preprocess_text(text: str) -> str:
    # 괄호 안 내용 삭제 (선택적)
    # text = re.sub(r"\([^)]*\)", "", text)

    # 특수문자 제거 (한글, 영문, 숫자, 일부 문장부호만 유지)
    text = re.sub(r"[^가-힣a-zA-Z0-9.,!?~\"'\s]", "", text)

    # 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    return text
