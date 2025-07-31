import re
import json
import os
from abc import ABC, abstractmethod
from glob import glob

class TextExtractor(ABC):
    """텍스트 추출 추상 클래스"""

    @abstractmethod
    def extract(self, source: str) -> tuple[list[str], list[dict]]:
        """텍스트와 메타데이터 추출"""
        pass

    @abstractmethod
    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        pass
