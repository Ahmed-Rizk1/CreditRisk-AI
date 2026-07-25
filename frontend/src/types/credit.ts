export interface CreditApplicantInput {
  LIMIT_BAL: number;
  SEX: number;
  EDUCATION: number;
  MARRIAGE: number;
  AGE: number;
  PAY_0: number;
  PAY_2: number;
  PAY_3: number;
  PAY_4: number;
  PAY_5: number;
  PAY_6: number;
  BILL_AMT1: number;
  BILL_AMT2: number;
  BILL_AMT3: number;
  BILL_AMT4: number;
  BILL_AMT5: number;
  BILL_AMT6: number;
  PAY_AMT1: number;
  PAY_AMT2: number;
  PAY_AMT3: number;
  PAY_AMT4: number;
  PAY_AMT5: number;
  PAY_AMT6: number;
}

export interface FeatureContribution {
  feature: string;
  feature_value: any;
  shap_value: number;
  impact: 'INCREASES_RISK' | 'REDUCES_RISK';
}

export interface SHAPExplanation {
  base_value: number;
  feature_contributions: FeatureContribution[];
  top_risk_drivers: FeatureContribution[];
  top_risk_reducers: FeatureContribution[];
}

export interface PredictionResponse {
  applicant_id: string;
  risk_score: number; // e.g. 14.5 (%)
  default_probability: number; // 0.0 to 1.0
  risk_category: 'LOW_RISK' | 'MEDIUM_RISK' | 'HIGH_RISK';
  decision: 'APPROVED' | 'MANUAL_REVIEW' | 'DECLINED';
  recommended_tier: string;
  adverse_action_reasons: string[];
  shap_explanation: SHAPExplanation;
  latency_ms: number;
}

export interface BatchPredictionSummary {
  total_applicants: number;
  approved_count: number;
  review_count: number;
  declined_count: number;
  average_risk_score: number;
}

export interface BatchPredictionResponse {
  summary: BatchPredictionSummary;
  predictions: PredictionResponse[];
}

export interface DriftColumnDetail {
  column_name: string;
  drift_detected: boolean;
  p_value: number;
  ks_statistic: number;
  threshold: number;
}

export interface DriftReportResponse {
  dataset_drift: boolean;
  number_of_drifted_columns: number;
  total_columns: number;
  drift_share: number;
  drift_by_columns: Record<string, DriftColumnDetail>;
}

export interface ModelInfoResponse {
  model_name: string;
  version: string;
  stage: string;
  metrics: Record<string, number>;
  registered_at?: string;
}
