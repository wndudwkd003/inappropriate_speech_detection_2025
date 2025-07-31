#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KURE 임베딩으로 가장 유사한 다른 answer를 찾아 rejected로 저장.
유사도가 낮으면 chosen의 일부 문장만 잘라서 rejected로 사용.
- 어떤 경우에도 데이터셋에 없던 새로운 문장을 추가하지 않음.

사용 예:
    python src/utils/dpo_data_make.py \
        --input data/raw/dev.json \
        --output data/processed/dev_dpo.json \
        --model nlpai-lab/KURE-v1 \
        --top_k 5 \
        --sim_threshold 0.60 \
        --keep_sents 1
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


def load_data(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_index(embs_norm: np.ndarray):
    dim = embs_norm.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs_norm)
    return index


def split_sentences(text: str):
    """
    문장 끝을 . ? ! (뒤에 따옴표 가능)로 간주하여 문장을 추출.
    새 문장 추가 없이 원문 일부만 반환.
    """
    pattern = re.compile(r'.+?(?:[.?!]["”]?|\Z)', flags=re.S)
    sents = [s.strip() for s in pattern.findall(text) if s.strip()]
    return sents


def truncate_answer(answer: str, keep_sents: int):
    """answer에서 앞부분 일부 문장만 남김. 새로운 문장 추가 금지."""
    sents = split_sentences(answer)
    if not sents:
        # 문장 분리가 안 되면 전체 문장을 반환 (추가 문장 금지)
        return answer
    # 요청한 개수만큼 잘라서 그대로 반환
    truncated = " ".join(sents[:keep_sents])
    return truncated


def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_data(input_path)
    prompts = [item["input"]["question"] for item in data]
    answers = [item["output"]["answer"] for item in data]
    ids = [item.get("id", str(i)) for i, item in enumerate(data)]

    # 1) 임베딩
    model = SentenceTransformer(args.model)
    embs = model.encode(answers, convert_to_numpy=True, show_progress_bar=True)
    embs_norm = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    # 2) 인덱스 생성
    index = build_index(embs_norm)

    # 3) 검색
    k = args.top_k
    scores, idxs = index.search(embs_norm, k)

    saved = 0
    pairs = []

    for i, (score_row, idx_row) in enumerate(zip(scores, idxs)):
        rejected_text = None
        rejected_from_id = None
        best_score = None

        # 자기 자신 제외하고 유사도 높은 순으로 탐색
        for j_idx, j in enumerate(idx_row):
            if j == i:
                continue
            sim = float(score_row[j_idx])
            if sim >= args.sim_threshold:
                # 다른 원본 데이터의 answer 그대로 사용
                rejected_text = answers[j]
                rejected_from_id = ids[j]
                best_score = sim
                break

        # fallback: 유사한 answer 없음 → chosen 일부 문장만 사용
        if rejected_text is None:
            rejected_text = truncate_answer(answers[i], args.keep_sents)
            rejected_from_id = ids[i]
            best_score = -1.0

        pair_obj = {
            "prompt": prompts[i],
            "chosen": answers[i],
            "rejected": rejected_text,
            "meta": {
                "chosen_id": ids[i],
                "rejected_from_id": rejected_from_id,
                "cosine": best_score
            }
        }
        pairs.append(pair_obj)
        saved += 1

    # JSON 배열로 한 번에 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved {saved} pairs to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/train.json")
    parser.add_argument("--output", type=str, default="data/processed/pair_list.json")
    parser.add_argument("--model", type=str, default="nlpai-lab/KURE-v1")
    parser.add_argument("--top_k", type=int, default=5, help="자기 자신 포함 K개 검색")
    parser.add_argument("--sim_threshold", type=float, default=0.60, help="유사도 임계값")
    parser.add_argument("--keep_sents", type=int, default=1, help="fallback 시 남길 문장 수")
    args_ = parser.parse_args()
    main(args_)

