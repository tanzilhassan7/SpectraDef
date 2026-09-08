export interface PredictionTop5Item {
  label: string;
  class_index: number;
  confidence: number;
}

export interface PredictionResponse {
  top1_label: string;
  top1_class_index: number;
  top1_confidence: number;
  top5: PredictionTop5Item[];
  latency_ms: number;
}

export interface AdversarialGenerateResponse {
  id: string;
  model: string;
  dataset: string;
  source_prediction: string;
  source_confidence: number;
  target_class: string;
  method: string;
  parameters: Record<string, any>;
  generation_status: 'success' | 'failed';
  status_reason?: string;
  adversarial_prediction: string;
  adversarial_confidence: number;
  created_at: string;
  purpose: string;
  diff_image_base64?: string;
}

export interface TransformedViewItem {
  name: string;
  strength: number;
  prediction: string;
  class_index: number;
  confidence: number;
  latency_ms: number;
  image_base64?: string;
}

export interface ConsistencyMetrics {
  prediction_agreement: number;
  prediction_switching_rate: number;
  dominant_prediction: string;
  dominant_class_index: number;
  mean_confidence: number;
  std_confidence: number;
  disagreement_with_raw: boolean;
  transformation_sensitivity: number;
}

export interface RiskEvaluation {
  risk_score: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  reasons: string[];
}

export interface MitigationDecision {
  final_decision: string;
  final_prediction: string;
  final_confidence: number;
  mitigation_action: string;
}

export interface ArgusAnalyzeResponse {
  shield: 'on' | 'off';
  raw_prediction: PredictionResponse;
  transformed_views?: TransformedViewItem[];
  consistency?: ConsistencyMetrics;
  risk?: RiskEvaluation;
  mitigation?: MitigationDecision;
  pipeline_stages: Record<string, string>;
}

export interface SweepDataPoint {
  strength: number;
  prediction: string;
  confidence: number;
  agreement: number;
  risk_contribution: number;
}

export interface RobustnessSweepResponse {
  transform_name: string;
  series: SweepDataPoint[];
}

export interface FixtureItem {
  id: string;
  name: string;
  description: string;
  ground_truth_label: string;
}

const API_BASE = '/api';

export async function getFixtures(): Promise<FixtureItem[]> {
  const res = await fetch(`${API_BASE}/fixtures`);
  if (!res.ok) throw new Error('Failed to fetch fixtures');
  return res.json();
}

export async function predictImage(formData: FormData): Promise<PredictionResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function generateAdversarial(formData: FormData): Promise<AdversarialGenerateResponse> {
  const res = await fetch(`${API_BASE}/adversarial/generate`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function analyzeArgus(formData: FormData): Promise<ArgusAnalyzeResponse> {
  const res = await fetch(`${API_BASE}/argus/analyze`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchRobustnessSweep(formData: FormData): Promise<RobustnessSweepResponse> {
  const res = await fetch(`${API_BASE}/robustness-sweep`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function runScriptedDemo(): Promise<any> {
  const res = await fetch(`${API_BASE}/demo/run`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
