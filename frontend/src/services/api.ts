import axios from 'axios';
import {
  CreditApplicantInput,
  PredictionResponse,
  BatchPredictionResponse,
  DriftReportResponse,
  ModelInfoResponse,
} from '../types/credit';

let workingBaseUrl: string | null = null;

export const getCandidateUrls = (): string[] => {
  const candidates: string[] = [];

  // 1. Build-time environment variable
  const envUrl = import.meta.env.VITE_API_BASE_URL || '';
  if (envUrl && envUrl.trim()) {
    const clean = envUrl.trim().replace(/\/$/, '');
    candidates.push(clean.endsWith('/api/v1') ? clean : `${clean}/api/v1`);
  }

  // 2. Render sub-domain auto-derivation
  if (typeof window !== 'undefined' && window.location.hostname.includes('.onrender.com')) {
    const currentHost = window.location.hostname;
    const baseName = currentHost.split('.')[0];
    
    if (baseName.includes('-frontend')) {
      candidates.push(`https://${currentHost.replace('-frontend', '-backend')}/api/v1`);
    }
    candidates.push(`https://${baseName}-backend.onrender.com/api/v1`);
    candidates.push(`https://creditrisk-backend.onrender.com/api/v1`);
  }

  // 3. Relative path for local dev proxy
  candidates.push('/api/v1');

  return Array.from(new Set(candidates));
};

export const getEffectiveApiBaseUrl = (): string => {
  if (workingBaseUrl) return workingBaseUrl;
  return getCandidateUrls()[0] || '/api/v1';
};

const apiClient = axios.create({
  timeout: 60000, // 60s timeout to allow Render Free Tier cold-starts
  headers: {
    'Content-Type': 'application/json',
  },
});

const executeWithFallback = async <T>(requestFn: (baseURL: string) => Promise<T>): Promise<T> => {
  const candidateUrls = getCandidateUrls();
  
  // If we already have a validated working URL, try it first
  if (workingBaseUrl) {
    try {
      return await requestFn(workingBaseUrl);
    } catch (err: any) {
      // If network error occurred, reset working URL and attempt candidates
      if (!err.response || err.code === 'ERR_NETWORK') {
        workingBaseUrl = null;
      } else {
        throw err; // HTTP 4xx/5xx errors returned by server, rethrow directly
      }
    }
  }

  let lastError: any = null;
  for (const url of candidateUrls) {
    try {
      const result = await requestFn(url);
      workingBaseUrl = url; // Save working candidate URL
      return result;
    } catch (err: any) {
      lastError = err;
      if (err.response) {
        // Server responded with an HTTP status code (4xx/5xx), so domain is reachable!
        workingBaseUrl = url;
        throw err;
      }
      // If network error (unreachable domain), continue to next candidate
    }
  }
  throw lastError;
};

export const api = {
  checkHealth: async (): Promise<{ status: string; service: string }> => {
    return executeWithFallback(async (baseURL) => {
      const healthUrl = baseURL.replace(/\/api\/v1$/, '') + '/health';
      const response = await axios.get<{ status: string; service: string }>(healthUrl, { timeout: 10000 });
      return response.data;
    });
  },

  getModelInfo: async (): Promise<ModelInfoResponse> => {
    return executeWithFallback(async (baseURL) => {
      const response = await apiClient.get<ModelInfoResponse>(`${baseURL}/model-info`);
      return response.data;
    });
  },

  predictSingle: async (input: CreditApplicantInput): Promise<PredictionResponse> => {
    return executeWithFallback(async (baseURL) => {
      const response = await apiClient.post<PredictionResponse>(`${baseURL}/predict`, input);
      return response.data;
    });
  },

  predictBatch: async (file: File): Promise<BatchPredictionResponse> => {
    return executeWithFallback(async (baseURL) => {
      const formData = new FormData();
      formData.append('file', file);
      const response = await apiClient.post<BatchPredictionResponse>(`${baseURL}/predict-batch`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return response.data;
    });
  },

  getDriftReport: async (): Promise<DriftReportResponse> => {
    return executeWithFallback(async (baseURL) => {
      const response = await apiClient.get<DriftReportResponse>(`${baseURL}/drift-report`);
      return response.data;
    });
  },
};

