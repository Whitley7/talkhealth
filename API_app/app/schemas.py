from pydantic import BaseModel
from typing import List, Dict
from typing import Optional

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    hates: Optional[List[str]] = []
    is_hate: bool
    is_ironic: bool
    hate_scores: Dict[str, float]
    irony_score: float