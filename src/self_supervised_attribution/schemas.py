from typing import List, Optional, Tuple, Union

from pydantic import BaseModel


class Answer(BaseModel):
    id: Union[str, int]
    text: str
    source: str


class ContextSource(BaseModel):
    id: Union[str, int]
    text: str
    source: str
    context_tokens: Tuple[int, int]


class Attribution(BaseModel):
    answer_tokens: Tuple[int, int]
    context_tokens: Tuple[int, int]


class AttributedAnswer(Answer):
    answer: Answer
    context: List[Union[str, int]]
    attribution: List[Attribution]


class Box(BaseModel):
    x: float
    y: float
    w: float
    h: float


class OCR(BaseModel):
    text: str
    box: Optional[Box] = None


class Page(BaseModel):
    number: int
    ocr: List[OCR] = list()


class Document(BaseModel):
    pages: List[Page] = list()
