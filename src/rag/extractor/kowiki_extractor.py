from src.rag.extractor.base_extractor import TextExtractor
import re, json, os
from glob import glob
from tqdm.auto import tqdm

class WikiExtractor(TextExtractor):
    """위키 텍스트 추출기"""

    def extract(self, source) -> tuple[list[str], list[dict]]:
        """위키에서 텍스트와 title 추출"""
        texts = []
        metadata = []

        # source가 리스트인지 확인하고 처리
        if isinstance(source, list):
            source_paths = source
        else:
            source_paths = [source]

        # 각 소스 경로에 대해 파일 처리
        for source_path in tqdm(source_paths, desc="Processing source files"):
            wiki_files = glob(os.path.join(source_path, "**/wiki_*"), recursive=True)
            for file_path in tqdm(wiki_files, desc=f"Processing {source_path}"):
                # 기존 파일 처리 로직 그대로 유지
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        doc = json.loads(line)
                        text = doc.get('text', '')
                        title = doc.get('title', '')
                        if not text or not title:
                            continue
                        cleaned_text = self.preprocess_text(text)
                        texts.append(cleaned_text)
                        metadata.append({
                            'title': title
                        })

        return texts, metadata

    def preprocess_text(self, text: str) -> str:
        """위키 텍스트 전처리"""
        # 템플릿 스타일 제거
        text = re.sub(r'<templatestyles[^>]*>', '', text)
        text = re.sub(r'</templatestyles>', '', text)

        # includeonly 태그 제거
        text = re.sub(r'<includeonly>.*?</includeonly>', '', text, flags=re.DOTALL)

        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)

        # 특수 위키 문법 제거
        text = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]+)\]\]', r'\1', text)
        text = re.sub(r'\{\{[^}]+\}\}', '', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&amp;', '&', text)

        # 비한글/영어/한자 문자 제거
        allowed_chars = r'[가-힣a-zA-Z0-9\u4e00-\u9fff\s\.,\(\)~\-:;!?\'"<>/]'
        text = ''.join(char if re.match(allowed_chars, char) else ' ' for char in text)

        # 연속된 공백 정리
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)

        return text.strip()
