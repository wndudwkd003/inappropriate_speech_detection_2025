import fitz, json, os, re
from glob import glob
from src.utils.rag_utils import preprocess_text


def extract_pdf(files: list[str], min_length: int = 50) -> list[str]:
    all_paragraphs = []

    for pdf_path in files:
        doc = fitz.open(pdf_path)

        buffer = ""
        for page in doc:
            text = page.get_text("text")
            lines = text.splitlines()

            for line in lines:
                line = preprocess_text(line)
                if not line:
                    continue

                if len(buffer) > 0:
                    buffer += " " + line
                else:
                    buffer = line

                # 한 문장이 너무 짧으면 다음 줄과 합치기 위해 계속 누적
                if len(buffer) < min_length:
                    continue
                else:
                    all_paragraphs.append(buffer.strip())
                    buffer = ""

        # 마지막 buffer 처리
        if buffer:
            all_paragraphs.append(buffer.strip())

    return all_paragraphs


def extract_wiki(base_dir: str, min_length: int = 50) -> list[str]:
    """
    위키 데이터에서 문단 추출 (extract_pdf와 동일한 인터페이스)

    Args:
        base_dir: 위키 데이터 디렉토리 경로
        min_length: 최소 문단 길이

    Returns:
        List[str]: 정제된 문단 리스트
    """
    paragraphs = []

    # 모든 wiki 파일 찾기
    wiki_files = glob(os.path.join(base_dir, "**/wiki_*"), recursive=True)

    for file_path in wiki_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                doc = json.loads(line)
                text = doc.get('text', '')

                if not text:
                    continue

                # 텍스트 정제
                cleaned_text = clean_wiki_text(text)

                # 최소 길이 체크
                if len(cleaned_text) < min_length:
                    continue

                paragraphs.append(cleaned_text)

    return paragraphs

def clean_wiki_text(text: str) -> str:
    """위키 텍스트 정제"""
    # 템플릿 스타일 제거
    text = re.sub(r'<templatestyles[^>]*>', '', text)
    text = re.sub(r'</templatestyles>', '', text)

    # includeonly 태그 제거
    text = re.sub(r'<includeonly>.*?</includeonly>', '', text, flags=re.DOTALL)

    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)

    # 특수 위키 문법 제거
    text = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]+)\]\]', r'\1', text)  # [[링크|텍스트]] → 텍스트
    text = re.sub(r'\{\{[^}]+\}\}', '', text)  # {{템플릿}} 제거
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&amp;', '&', text)

    # 비한글/영어/한자 문자 제거 (기본 특수문자 유지)
    allowed_chars = r'[가-힣a-zA-Z0-9\u4e00-\u9fff\s\.,\(\)~\-:;!?\'"<>/]'
    text = ''.join(char if re.match(allowed_chars, char) else ' ' for char in text)

    # 연속된 공백 정리
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)

    return text.strip()
