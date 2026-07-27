import React, { useState } from 'react';
import { CreditApplicantInput, PredictionResponse } from '../types/credit';
import { api } from '../services/api';
import { ShieldAlert, CheckCircle2, AlertTriangle, XCircle, ArrowRight, Activity, Info, Zap } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';

export const SingleAssessment: React.FC = () => {
  const [formData, setFormData] = useState<CreditApplicantInput>({
    LIMIT_BAL: 50000,
    SEX: 2,
    EDUCATION: 2,
    MARRIAGE: 1,
    AGE: 35,
    PAY_0: 0,
    PAY_2: 0,
    PAY_3: 0,
    PAY_4: 0,
    PAY_5: 0,
    PAY_6: 0,
    BILL_AMT1: 20000,
    BILL_AMT2: 19000,
    BILL_AMT3: 18000,
    BILL_AMT4: 17000,
    BILL_AMT5: 16000,
    BILL_AMT6: 15000,
    PAY_AMT1: 2000,
    PAY_AMT2: 2000,
    PAY_AMT3: 2000,
    PAY_AMT4: 2000,
    PAY_AMT5: 2000,
    PAY_AMT6: 2000,
  });

  const [loading, setLoading] = useState<boolean>(false);
  const [isWakingUp, setIsWakingUp] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: parseFloat(value) || 0,
    }));
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);
    setIsWakingUp(false);

    // If server cold starts, notify user after 3 seconds
    const wakeupTimer = setTimeout(() => {
      setIsWakingUp(true);
    }, 3000);

    try {
      const res = await api.predictSingle(formData);
      setResult(res);
    } catch (err: any) {
      console.error('Prediction error:', err);
      const errMsg = err?.response?.data?.detail 
        || err?.message 
        || 'Network error or backend timeout. Render Free Tier may be waking up.';
      setError(errMsg);
    } finally {
      clearTimeout(wakeupTimer);
      setIsWakingUp(false);
      setLoading(false);
    }
  };

  // Format SHAP data for Recharts Waterfall
  const shapData = result?.shap_explanation.feature_contributions
    ? result.shap_explanation.feature_contributions
        .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
        .slice(0, 8)
        .map(item => ({
          name: item.feature.replace('_', ' '),
          impact: Number((item.shap_value * 100).toFixed(2)),
          raw: item.feature_value,
          direction: item.impact,
        }))
    : [];

  return (
    <div style={{ padding: '32px 0' }}>
      <div style={{ display: 'grid', gridTemplateColumns: result ? '1fr 1.2fr' : '1fr', gap: '32px' }}>
        
        {/* Form Column */}
        <div className="glass-card" style={{ padding: '28px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '24px' }}>
            <Activity color="var(--accent-cyan)" size={22} />
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700 }}>Borrower Profile Input</h2>
          </div>

          <form onSubmit={handleSubmit}>
            {/* Section 1: Financial & Demographic */}
            <p style={{ fontSize: '0.85rem', color: 'var(--accent-cyan)', fontWeight: 600, marginBottom: '12px' }}>
              1. Financial & Demographic Attributes
            </p>
            <div className="grid-2" style={{ marginBottom: '20px' }}>
              <div>
                <label className="input-label">Credit Limit (NT$)</label>
                <input
                  type="number"
                  name="LIMIT_BAL"
                  value={formData.LIMIT_BAL}
                  onChange={handleChange}
                  className="input-field"
                  step="5000"
                />
              </div>

              <div>
                <label className="input-label">Age (Years)</label>
                <input
                  type="number"
                  name="AGE"
                  value={formData.AGE}
                  onChange={handleChange}
                  className="input-field"
                />
              </div>

              <div>
                <label className="input-label">Education Level</label>
                <select name="EDUCATION" value={formData.EDUCATION} onChange={handleChange} className="input-field">
                  <option value={1}>Graduate School</option>
                  <option value={2}>University Degree</option>
                  <option value={3}>High School</option>
                  <option value={4}>Others / Trade</option>
                </select>
              </div>

              <div>
                <label className="input-label">Marital Status</label>
                <select name="MARRIAGE" value={formData.MARRIAGE} onChange={handleChange} className="input-field">
                  <option value={1}>Married</option>
                  <option value={2}>Single</option>
                  <option value={3}>Others</option>
                </select>
              </div>
            </div>

            {/* Section 2: Payment Repayment History */}
            <p style={{ fontSize: '0.85rem', color: 'var(--accent-cyan)', fontWeight: 600, marginBottom: '12px' }}>
              2. Repayment History (Past 3 Months)
            </p>
            <div className="grid-3" style={{ marginBottom: '20px' }}>
              <div>
                <label className="input-label">Sept Payment Status</label>
                <select name="PAY_0" value={formData.PAY_0} onChange={handleChange} className="input-field">
                  <option value={-1}>Pay Duly (-1)</option>
                  <option value={0}>Revolving Credit (0)</option>
                  <option value={1}>1 Month Delay</option>
                  <option value={2}>2 Months Delay</option>
                  <option value={3}>3+ Months Delay</option>
                </select>
              </div>

              <div>
                <label className="input-label">Aug Payment Status</label>
                <select name="PAY_2" value={formData.PAY_2} onChange={handleChange} className="input-field">
                  <option value={-1}>Pay Duly (-1)</option>
                  <option value={0}>Revolving Credit (0)</option>
                  <option value={1}>1 Month Delay</option>
                  <option value={2}>2 Months Delay</option>
                  <option value={3}>3+ Months Delay</option>
                </select>
              </div>

              <div>
                <label className="input-label">July Payment Status</label>
                <select name="PAY_3" value={formData.PAY_3} onChange={handleChange} className="input-field">
                  <option value={-1}>Pay Duly (-1)</option>
                  <option value={0}>Revolving Credit (0)</option>
                  <option value={1}>1 Month Delay</option>
                  <option value={2}>2 Months Delay</option>
                  <option value={3}>3+ Months Delay</option>
                </select>
              </div>
            </div>

            {/* Section 3: Bill Statements & Payments */}
            <p style={{ fontSize: '0.85rem', color: 'var(--accent-cyan)', fontWeight: 600, marginBottom: '12px' }}>
              3. Recent Bill Statements & Payments
            </p>
            <div className="grid-2" style={{ marginBottom: '24px' }}>
              <div>
                <label className="input-label">Sept Bill Amount (NT$)</label>
                <input
                  type="number"
                  name="BILL_AMT1"
                  value={formData.BILL_AMT1}
                  onChange={handleChange}
                  className="input-field"
                />
              </div>

              <div>
                <label className="input-label">Sept Paid Amount (NT$)</label>
                <input
                  type="number"
                  name="PAY_AMT1"
                  value={formData.PAY_AMT1}
                  onChange={handleChange}
                  className="input-field"
                />
              </div>
            </div>

            <button type="submit" className="btn-primary" style={{ width: '100%', justifyContent: 'center' }} disabled={loading}>
              {loading ? <Zap size={18} className="animate-spin" /> : <ShieldAlert size={18} />}
              {loading ? (isWakingUp ? 'Waking up Render backend instance...' : 'Calculating Risk & SHAP Attributions...') : 'Evaluate Credit Risk & Explain'}
            </button>
          </form>

          {isWakingUp && (
            <div style={{
              marginTop: '16px',
              padding: '12px 16px',
              borderRadius: '10px',
              background: 'rgba(245, 158, 11, 0.1)',
              border: '1px solid var(--status-warning)',
              color: 'var(--status-warning)',
              fontSize: '0.85rem',
              display: 'flex',
              alignItems: 'center',
              gap: '10px'
            }}>
              <Zap size={18} className="animate-spin" />
              <div>
                <strong>Backend spin-up detected:</strong> Render Free Tier service is spinning up from cold storage (~30-50s). Please stay on this page!
              </div>
            </div>
          )}

          {error && (
            <div style={{
              marginTop: '16px',
              padding: '16px',
              borderRadius: '12px',
              background: 'rgba(244, 63, 94, 0.1)',
              border: '1px solid var(--status-danger)',
              color: '#fecdd3'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontWeight: 700, marginBottom: '6px', color: 'var(--status-danger)' }}>
                <XCircle size={20} /> Evaluation Request Failed
              </div>
              <p style={{ fontSize: '0.88rem', margin: 0, marginBottom: '12px', lineHeight: 1.4 }}>
                {error}
              </p>
              <button
                type="button"
                onClick={() => handleSubmit()}
                className="btn-secondary"
                style={{ padding: '6px 14px', fontSize: '0.82rem', borderColor: 'var(--status-danger)', color: '#fff' }}
              >
                Retry Request
              </button>
            </div>
          )}
        </div>

        {/* Prediction Results & SHAP Explanation Column */}
        {result && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* Top Risk Score Card */}
            <div className="glass-card" style={{ padding: '24px', borderLeft: `6px solid ${
              result.decision === 'APPROVED' ? 'var(--status-success)' :
              result.decision === 'MANUAL_REVIEW' ? 'var(--status-warning)' : 'var(--status-danger)'
            }` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
                <div>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Application ID: {result.applicant_id}
                  </span>
                  <h3 style={{ fontSize: '1.8rem', fontWeight: 800, marginTop: '4px' }}>
                    {result.risk_score}% <span style={{ fontSize: '1rem', fontWeight: 400, color: 'var(--text-muted)' }}>Default Probability</span>
                  </h3>
                </div>

                {/* Decision Badge */}
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <span className={
                    result.decision === 'APPROVED' ? 'badge-success' :
                    result.decision === 'MANUAL_REVIEW' ? 'badge-warning' : 'badge-danger'
                  } style={{ padding: '8px 16px', borderRadius: '12px', fontWeight: 700, fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    {result.decision === 'APPROVED' && <CheckCircle2 size={18} />}
                    {result.decision === 'MANUAL_REVIEW' && <AlertTriangle size={18} />}
                    {result.decision === 'DECLINED' && <XCircle size={18} />}
                    {result.decision}
                  </span>
                </div>
              </div>

              {/* Recommendation Tier */}
              <div style={{ marginTop: '16px', paddingTop: '16px', borderTop: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', fontSize: '0.88rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Recommended Product Tier:</span>
                <strong style={{ color: 'var(--accent-cyan)' }}>{result.recommended_tier}</strong>
              </div>
            </div>

            {/* SHAP Waterfall Chart */}
            <div className="glass-card" style={{ padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <h4 style={{ fontSize: '1.05rem', fontWeight: 700 }}>
                  SHAP Visual Feature Attributions
                </h4>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Info size={14} /> Red increases risk, Green lowers risk
                </span>
              </div>

              <div style={{ height: '240px', width: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={shapData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                    <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                    <YAxis type="category" dataKey="name" tick={{ fill: '#f8fafc', fontSize: 12 }} width={120} />
                    <Tooltip
                      contentStyle={{ background: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#fff' }}
                      formatter={(val: number) => [`${val > 0 ? '+' : ''}${val}% Impact`, 'SHAP Score']}
                    />
                    <ReferenceLine x={0} stroke="#475569" />
                    <Bar dataKey="impact">
                      {shapData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.impact > 0 ? '#f43f5e' : '#10b981'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Adverse Action Notice Reasons */}
            {result.adverse_action_reasons && result.adverse_action_reasons.length > 0 && (
              <div className="glass-card" style={{ padding: '20px', borderLeft: '4px solid var(--accent-indigo)' }}>
                <h4 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: '10px', color: 'var(--accent-indigo)' }}>
                  Regulatory Adverse Action Notice Reasons
                </h4>
                <ul style={{ paddingLeft: '20px', fontSize: '0.88rem', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  {result.adverse_action_reasons.map((reason, idx) => (
                    <li key={idx}>{reason}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
