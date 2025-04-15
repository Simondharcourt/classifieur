export interface ClassificationResponse {
  category: string;
  confidence: number;
  explanation: string;
}

export interface BatchClassificationResponse {
  results: ClassificationResponse[];
}

export interface CategorySuggestionResponse {
  categories: string[];
}

export interface ValidationSample {
  text: string;
  assigned_category: string;
  confidence: number;
}

export interface ValidationRequest {
  samples: ValidationSample[];
  current_categories: string[];
  text_columns: string[];
}

export interface ValidationResponse {
  validation_report: string;
  accuracy_score?: number;
  misclassifications?: Array<{
    text: string;
    current_category: string;
  }>;
  suggested_improvements?: string[];
}

export interface ImprovementRequest {
  df: Record<string, any>;
  validation_report: string;
  text_columns: string[];
  categories: string;
  classifier_type: string;
  show_explanations: boolean;
  file_path: string;
}

export interface ImprovementResponse {
  improved_df: Record<string, any>;
  new_validation_report: string;
  success: boolean;
  updated_categories: string[];
}

export interface ModelInfoResponse {
  model_name: string;
  model_version: string;
  max_tokens: number;
  temperature: number;
}

export interface HealthResponse {
  status: string;
  model_ready: boolean;
  api_key_configured: boolean;
} 