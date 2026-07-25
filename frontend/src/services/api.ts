import axios from 'axios';
import {
  CreditApplicantInput,
  PredictionResponse,
  BatchPredictionResponse,
  DriftReportResponse,
  ModelInfoResponse,
} from '../types/credit';

const RAW_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';
const API_BASE_URL = RAW_BASE_URL 
  ? (RAW_BASE_URL.endsWith('/api/v1') ? RAW_BASE_URL : `${RAW_BASE_URL.replace(/\/$/, '')}/api/v1`)
  : '/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  getModelInfo: async (): Promise<ModelInfoResponse> => {
    const response = await apiClient.get<ModelInfoResponse>('/model-info');
    return response.data;
  },

  predictSingle: async (input: CreditApplicantInput): Promise<PredictionResponse> => {
    const response = await apiClient.post<PredictionResponse>('/predict', input);
    return response.data;
  },

  predictBatch: async (file: File): Promise<BatchPredictionResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await apiClient.post<BatchPredictionResponse>('/predict-batch', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  getDriftReport: async (): Promise<DriftReportResponse> => {
    const response = await apiClient.get<DriftReportResponse>('/drift-report');
    return response.data;
  },
};
