# Frontend Guidelines - CreditRisk AI

This document provides instructions for AI agents working in the `frontend/` directory.

## Frontend Stack
- **Framework:** React SPA + Vite + TypeScript
- **Icons:** Lucide-React
- **Charts & Visualizations:** Recharts / Chart.js
- **HTTP Client:** Axios (configured in `frontend/src/services/api.ts`)

## Design Principles
1. **Premium Aesthetic:** Modern dark/light theme, sleek glassmorphism, responsive cards, crisp typography (Inter/Roboto), and interactive data charts.
2. **Clear Feedback:** Always show loading spinners during model inference or batch file processing.
3. **Interactive SHAP Visualizer:** Render clear, color-coded feature contribution bars (green for risk-lowering factors, red for risk-increasing factors).
4. **No Hardcoded API URLs:** Read backend URL from environment variables (`import.meta.env.VITE_API_BASE_URL`).

## Execution
```powershell
npm install
npm run dev
```
