# compare_two_jsons.py
import json
import math
import filecmp
from collections import Counter
from typing import Any, List, Set

# ----------------------------------------------------------------------
# 하드코딩된 비교 대상 경로
# ----------------------------------------------------------------------
LEFT = r"output/2025-07-27_10-30-56_kakaocorp_kanana-1.5-8b-instruct-2505_r_128_ra_256_rd_0.2_float16_sft/test_result/test_results_896cb8a9_2025-07-27_12-54-01_.json"
RIGHT = r"output/2025-07-27_10-30-56_kakaocorp_kanana-1.5-8b-instruct-2505_r_128_ra_256_rd_0.2_float16_sft/test_result/test_results_896cb8a9_2025-07-31_06-36-16.json"

# ----------------------------------------------------------------------
# 비교 세부 설정(필요 시만 수정)
# ----------------------------------------------------------------------
IGNORE_KEYS: Set[str] = set()   # 예: {"timestamp", "run_id", "started_at", "finished_at"}
FLOAT_TOL: float = 1e-9         # 부동소수점 허용오차
LIST_ORDER: str = "strict"       # "strict"(순서 고려) 또는 "set"(순서 무시, 다중집합 비교)
MAX_DIFFS: int = 200             # 최대 차이 리포트 개수

# ----------------------------------------------------------------------
# 유틸 함수들
# ----------------------------------------------------------------------
def load_json_or_jsonl(path: str) -> Any:
    """일반 JSON을 우선 시도하고, 실패하면 JSONL(행별 JSON)로 로드합니다."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        objs = []
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if not s:
                continue
            try:
                objs.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 파싱 오류: {path}:{i}번째 줄 - {e}") from e
        return objs

def canonical(obj: Any) -> str:
    """리스트 원소를 표준 문자열로 직렬화(키 정렬)하여 다중집합 비교에 사용."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)

def compare_values(a: Any, b: Any, path: str, diffs: List[str],
                   ignore_keys: Set[str], float_tol: float,
                   list_order: str, max_diffs: int) -> None:
    if len(diffs) >= max_diffs:
        return

    # 타입이 다르면 바로 차이
    if type(a) != type(b):
        diffs.append(f"{path}: 타입 불일치 {type(a).__name__} != {type(b).__name__}")
        return

    # 딕셔너리 비교
    if isinstance(a, dict):
        a_keys = set(a.keys()) - ignore_keys
        b_keys = set(b.keys()) - ignore_keys

        # 한쪽에만 있는 키
        for k in sorted(a_keys - b_keys):
            if len(diffs) >= max_diffs: return
            diffs.append(f"{path}.{k}: 오른쪽에 없음")
        for k in sorted(b_keys - a_keys):
            if len(diffs) >= max_diffs: return
            diffs.append(f"{path}.{k}: 왼쪽에 없음")

        # 공통 키는 재귀 비교
        for k in sorted(a_keys & b_keys):
            if len(diffs) >= max_diffs: return
            compare_values(a[k], b[k], f"{path}.{k}", diffs, ignore_keys, float_tol, list_order, max_diffs)
        return

    # 리스트 비교
    if isinstance(a, list):
        if list_order == "strict":
            if len(a) != len(b):
                diffs.append(f"{path}: 리스트 길이 불일치 {len(a)} != {len(b)}")
                if len(diffs) >= max_diffs: return
            # 같은 인덱스끼리 비교
            for i in range(min(len(a), len(b))):
                if len(diffs) >= max_diffs: return
                compare_values(a[i], b[i], f"{path}[{i}]", diffs, ignore_keys, float_tol, list_order, max_diffs)
        elif list_order in ("set", "bag", "multiset"):
            ca = Counter(canonical(x) for x in a)
            cb = Counter(canonical(x) for x in b)
            # 왼쪽에만 있는 항목
            for item, cnt in (ca - cb).items():
                if len(diffs) >= max_diffs: return
                diffs.append(f"{path}: 왼쪽에만 {cnt}개 존재 -> {item}")
            # 오른쪽에만 있는 항목
            for item, cnt in (cb - ca).items():
                if len(diffs) >= max_diffs: return
                diffs.append(f"{path}: 오른쪽에만 {cnt}개 존재 -> {item}")
        else:
            raise ValueError(f"LIST_ORDER 값이 올바르지 않습니다: {list_order}")
        return

    # float 비교(허용오차)
    if isinstance(a, float):
        if not math.isclose(a, b, rel_tol=float_tol, abs_tol=float_tol):
            diffs.append(f"{path}: float 불일치 {a} != {b} (tol={float_tol})")
        return

    # 스칼라 비교
    if a != b:
        if isinstance(a, str) and ("\n" in a or len(a) > 120 or "\n" in str(b) or len(str(b)) > 120):
            diffs.append(f"{path}: 문자열 불일치 (길이 {len(a)} != {len(b)})")
        else:
            diffs.append(f"{path}: 값 불일치 {a!r} != {b!r}")

def main():
    # 1) 바이트 단위 동일 여부
    try:
        byte_equal = filecmp.cmp(LEFT, RIGHT, shallow=False)
    except FileNotFoundError as e:
        print(f"[오류] 파일을 찾을 수 없습니다: {e}")
        return

    # 2) JSON 의미 비교
    try:
        left_obj = load_json_or_jsonl(LEFT)
        right_obj = load_json_or_jsonl(RIGHT)
    except Exception as e:
        print(f"[오류] JSON 로드/파싱 실패: {e}")
        return

    diffs: List[str] = []
    compare_values(
        left_obj, right_obj, path="$", diffs=diffs,
        ignore_keys=IGNORE_KEYS,
        float_tol=FLOAT_TOL,
        list_order=LIST_ORDER,
        max_diffs=MAX_DIFFS,
    )

    # 결과 출력
    print("=== 비교 결과 요약 ===")
    print(f"바이트 동일: {byte_equal}")
    print(f"JSON 의미 동일: {len(diffs) == 0}")
    print(f"차이 개수(보고된 것): {len(diffs)} / 최대 출력 {MAX_DIFFS}")

    if diffs:
        print("\n=== 차이 상세(상위 일부) ===")
        for d in diffs:
            print(d)
        if len(diffs) >= MAX_DIFFS:
            print("... 생략됨(더 많은 차이가 존재할 수 있음)")

if __name__ == "__main__":
    main()
