import axios from 'axios';
import {
  ClassificationResponse,
  BatchClassificationResponse,
  CategorySuggestionResponse,
  ValidationRequest,
  ValidationResponse,
  ImprovementRequest,
  ImprovementResponse,
  ModelInfoResponse,
  HealthResponse
} from '../types/api';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const healthCheck = async (): Promise<HealthResponse> => {
  const response = await api.get<HealthResponse>('/health');
  return response.data;
};

export const getModelInfo = async (): Promise<ModelInfoResponse> => {
  const response = await api.get<ModelInfoResponse>('/model-info');
  return response.data;
};

export const classifyText = async (text: string, categories?: string[]): Promise<ClassificationResponse> => {
  const response = await api.post<ClassificationResponse>('/classify', {
    text,
    categories,
  });
  return response.data;
};

export const classifyBatch = async (texts: string[], categories?: string[]): Promise<BatchClassificationResponse> => {
  const response = await api.post<BatchClassificationResponse>('/classify-batch', {
    texts,
    categories,
  });
  return response.data;
};

export const suggestCategories = async (texts: string[]): Promise<CategorySuggestionResponse> => {
  const response = await api.post<CategorySuggestionResponse>('/suggest-categories', texts);
  return response.data;
};

export const validateClassifications = async (request: ValidationRequest): Promise<ValidationResponse> => {
  const response = await api.post<ValidationResponse>('/validate', request);
  return response.data;
};

export const improveClassification = async (request: ImprovementRequest): Promise<ImprovementResponse> => {
  const response = await api.post<ImprovementResponse>('/improve-classification', request);
  return response.data;
}; 