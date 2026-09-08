from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PredictionTop5Item(BaseModel):
    label: str
    class_index: int
    confidence: float

class PredictionResponse(BaseModel):
    top1_label: str
    top1_class_index: int
    top1_confidence: float
    top5: List[PredictionTop5Item]
    latency_ms: float

class AdversarialGenerateRequest(BaseModel):
    fixture_id: Optional[str] = None
    method: str = "iterative_target"  # "fgsm", "one_step_targeted", "iterative", "iterative_target"
    target_class: int = 9  # default Ostrich
    strength_preset: str = "moderate"  # "controlled", "moderate", "strong"
    iterations_preset: int = 10

class AdversarialGenerateResponse(BaseModel):
    id: str
    model: str = "InceptionV3"
    dataset: str = "ImageNet"
    source_prediction: str
    source_confidence: float
    target_class: str
    method: str
    parameters: Dict[str, Any]
    generation_status: str  # "success" or "failed"
    status_reason: Optional[str] = None
    adversarial_prediction: str
    adversarial_confidence: float
    created_at: str
    purpose: str = "ARGUS controlled robustness evaluation"
    diff_image_base64: Optional[str] = None

class AdversarialVerifyRequest(BaseModel):
    artifact_id: str

class TransformedViewItem(BaseModel):
    name: str
    strength: float
    prediction: str
    class_index: int
    confidence: float
    latency_ms: float
    image_base64: Optional[str] = None

class ConsistencyMetrics(BaseModel):
    prediction_agreement: float
    prediction_switching_rate: int
    dominant_prediction: str
    dominant_class_index: int
    mean_confidence: float
    std_confidence: float
    disagreement_with_raw: bool
    transformation_sensitivity: float

class RiskEvaluation(BaseModel):
    risk_score: float  # 0 to 100
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    reasons: List[str]

class MitigationDecision(BaseModel):
    final_decision: str  # "ACCEPTED", "CONSENSUS_RECOVERED", "ABSTAIN"
    final_prediction: str
    final_confidence: float
    mitigation_action: str

class ArgusAnalyzeRequest(BaseModel):
    artifact_id: Optional[str] = None
    shield: str = "on"  # "on" or "off"

class ArgusAnalyzeResponse(BaseModel):
    shield: str
    raw_prediction: PredictionResponse
    transformed_views: Optional[List[TransformedViewItem]] = None
    consistency: Optional[ConsistencyMetrics] = None
    risk: Optional[RiskEvaluation] = None
    mitigation: Optional[MitigationDecision] = None
    pipeline_stages: Dict[str, str]

class SweepDataPoint(BaseModel):
    strength: float
    prediction: str
    confidence: float
    agreement: float
    risk_contribution: float

class RobustnessSweepResponse(BaseModel):
    transform_name: str
    series: List[SweepDataPoint]

class FixtureItem(BaseModel):
    id: str
    name: str
    description: str
    ground_truth_label: str
