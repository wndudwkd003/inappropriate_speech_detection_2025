from src.rag.extractor.base_extractor import TextExtractor
import re
import fitz
from tqdm.auto import tqdm

class PDFExtractor(TextExtractor):
    """PDF 텍스트 추출기 - <제목 - 부제> 형식 파싱"""

    def extract(self, source: str) -> tuple[list[str], list[dict]]:
        """PDF에서 텍스트와 title 추출"""
        texts = []
        metadata = []

        # PDF 파일 읽기
        for pdf_path in source:  # source는 파일 경로 리스트
            doc = fitz.open(pdf_path)
            full_text = ""

            # 전체 텍스트 추출
            for page in doc:
                page_text = page.get_text("text")
                full_text += page_text + "\n"

            doc.close()

            # <제목 - 부제> 패턴으로 분할
            sections = self._parse_sections(full_text)

            # 각 섹션 처리
            for title, content in tqdm(sections, desc=f"Processing {pdf_path}"):
                cleaned_content = self.preprocess_text(content)

                texts.append(cleaned_content)
                metadata.append({
                    'title': title
                })

        return texts, metadata

    def _parse_sections(self, text: str) -> list[tuple[str, str]]:
        """<제목 - 부제> 형식으로 섹션 분할"""
        sections = []

        # <제목 - 부제> 패턴 찾기
        pattern = r'<([^>]+)>'
        matches = list(re.finditer(pattern, text))

        if not matches:
            # 패턴이 없으면 전체를 하나의 섹션으로
            return [("Document", text)]

        for i, match in enumerate(matches):
            # 제목 추출 (< > 내부)
            title_text = match.group(1).strip()

            # 제목에서 ' - '로 분리된 부분 처리
            if ' - ' in title_text:
                title = title_text  # 전체를 제목으로 사용
            else:
                title = title_text

            # 내용 추출 (현재 제목부터 다음 제목 전까지)
            start_pos = match.end()
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)

            content = text[start_pos:end_pos].strip()

            if content:  # 내용이 있는 경우만 추가
                sections.append((title, content))

        return sections

    def preprocess_text(self, text: str) -> str:
        """PDF 텍스트 전처리"""
        # 특수문자 제거 (한글, 영문, 숫자, 한자, 일본어, 일부 문장부호만 유지)
        # 한자: \u4e00-\u9fff
        # 일본어 히라가나: \u3040-\u309f
        # 일본어 가타카나: \u30a0-\u30ff
        text = re.sub(r"[^가-힣a-zA-Z0-9\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff.,!?~\"'\s]", "", text)
        # 공백 정리
        text = re.sub(r"\s+", " ", text).strip()
        return text
