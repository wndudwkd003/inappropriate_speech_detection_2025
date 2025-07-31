from src.rag.extractor.base_extractor import TextExtractor
from src.rag.extractor.pdf_extractor import PDFExtractor
from src.rag.extractor.kowiki_extractor import WikiExtractor


class ExtractorFactory:
    """추출기 팩토리"""

    _extractors = {
        'pdf': PDFExtractor,
        'kowiki': WikiExtractor
    }

    @classmethod
    def get_extractor(cls, style: str) -> TextExtractor:
        """스타일에 맞는 추출기 반환"""
        if style not in cls._extractors:
            raise ValueError(f"Unsupported style: {style}. Available: {list(cls._extractors.keys())}")

        return cls._extractors[style]()
