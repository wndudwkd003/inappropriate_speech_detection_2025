from tqdm.auto import tqdm

def chunk_texts(
    texts: list[str],
    metadata: list[dict],
    chunk_size: int,
    chunk_overlap: int,
    min_last_chunk_ratio: float = 0.2
) -> tuple[list[str], list[dict]]:

    """텍스트를 청크로 분할하면서 메타데이터(title) 유지"""
    chunks = []
    chunk_metadata = []

    pair = zip(texts, metadata)
    for text, meta in tqdm(pair, desc="Chunking texts"):
        title = meta['title']
        words = text.split()

        if len(words) <= chunk_size:
            chunks.append(text)
            chunk_metadata.append({
                'title': title,
                'chunk_idx': 0
            })
            continue

        # 긴 텍스트는 청킹
        doc_chunks = []
        doc_chunk_metadata = []
        start = 0
        chunk_idx = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            doc_chunks.append(chunk)
            doc_chunk_metadata.append({
                'title': title,
                'chunk_idx': chunk_idx
            })
            chunk_idx += 1

            if end == len(words):
                break
            start += chunk_size - chunk_overlap

        # 마지막 청크가 너무 작으면 이전 청크와 병합
        if len(doc_chunks) > 1:
            last_chunk_words = doc_chunks[-1].split()
            min_size = int(chunk_size * min_last_chunk_ratio)

            if len(last_chunk_words) < min_size:
                # 마지막 청크를 이전 청크와 병합
                prev_chunk_words = doc_chunks[-2].split()
                merged_chunk = " ".join(prev_chunk_words + last_chunk_words)

                # 병합된 청크로 교체
                doc_chunks[-2] = merged_chunk
                doc_chunks.pop()
                doc_chunk_metadata.pop()

        chunks.extend(doc_chunks)
        chunk_metadata.extend(doc_chunk_metadata)

    return chunks, chunk_metadata
