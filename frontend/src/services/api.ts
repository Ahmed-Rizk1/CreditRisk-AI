import axios from 'axios';
import {
  CreditApplicantInput,
  PredictionResponse,
  BatchPredictionResponse,
  DriftReportResponse,
  ModelInfoResponse,
} from '../types/credit';

export const getEffectiveApiBaseUrl = (): string => {
  // 1. Check user custom override in localStorage
  const savedUrl = typeof window !== 'undefined' ? localStorage.getItem('CREDITRISK_API_URL') : null;
  if (savedUrl && savedUrl.trim()) {
    const clean = savedUrl.trim().replace(/\/$/, '');
    return clean.endsWith('/api/v1') ? clean : `${clean}/api/v1`;
  }

  // 2. Check build-time env var
  const envUrl = import.meta.env.VITE_API_BASE_URL || '';
  if (envUrl && envUrl.trim()) {
    const clean = envUrl.trim().replace(/\/$/, '');
    return clean.endsWith('/api/v1') ? clean : `${clean}/api/v1`;
  }

  // 3. Render auto-domain derivation (if running on static site on render.com)
  if (typeof window !== 'undefined' && window.location.hostname.includes('.onrender.com')) {
    const currentHost = window.location.hostname;
    if (currentHost.includes('-frontend')) {
      const derivedBackendHost = currentHost.replace('-frontend', '-backend');
      return `https://${derivedBackendHost}/api/v1`;
    }
  }

  // 4. Default relative API path for local dev proxy
  return '/api/v1';
};

const apiClient = axios.create({
  timeout: 60000, // 60s timeout to allow Render Free Tier cold-starts
  headers: {
    'Content-Type': 'application/json',
  },
});

// Interceptor to dynamically update baseURL per request
apiClient.interceptors.request.use((config) => {
  config.baseURL = getEffectiveApiBaseUrl();
  return config;
});

export const api = {
  checkHealth: async (): Promise<{ status: string; service: string }> => {
    const baseUrl = getEffectiveApiBaseUrl().replace(/\/api\/v1$/, '');
    const response = await axios.get<{ status: string; service: string }>(`${baseUrl}/health`, { timeout: 10000 });
    return response.data;
  },

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

