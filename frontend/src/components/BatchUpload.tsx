import React, { useState } from 'react';
import { api } from '../services/api';
import { BatchPredictionResponse } from '../types/credit';
import { UploadCloud, FileSpreadsheet, CheckCircle2, AlertTriangle, XCircle, Download, RefreshCw } from 'lucide-react';

export const BatchUpload: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [batchResult, setBatchResult] = useState<BatchPredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    try {
      const res = await api.predictBatch(file);
      setBatchResult(res);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to process batch CSV file.');
    } finally {
      setLoading(false);
    }
  };

  const handleExportCSV = () => {
    if (!batchResult) return;
    const headers = ['Applicant ID', 'Risk Score (%)', 'Default Probability', 'Risk Category', 'Decision', 'Recommended Tier'];
    const rows = batchResult.predictions.map(p => [
      p.applicant_id,
      p.risk_score,
      p.default_probability,
      p.risk_category,
      p.decision,
      `"${p.recommended_tier}"`
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `CreditRisk_Batch_Export_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div style={{ padding: '32px 0' }}>
      {/* Drag & Drop Upload Zone */}
      <div className="glass-card" style={{ padding: '36px', textAlign: 'center', marginBottom: '32px' }}>
        <div style={{
          width: '64px',
          height: '64px',
          borderRadius: '50%',
          background: 'rgba(99, 102, 241, 0.15)',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginBottom: '16px'
        }}>
          <UploadCloud size={32} color="var(--accent-cyan)" />
        </div>

        <h2 style={{ fontSize: '1.4rem', fontWeight: 700, marginBottom: '8px' }}>
          Bulk Portfolio CSV Scoring
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '24px', maxWidth: '500px', margin: '0 auto 24px auto' }}>
          Upload a CSV file containing multiple applicant profiles to run parallel vectorized risk scoring and decision classification.
        </p>

        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            id="csv-file-input"
            style={{ display: 'none' }}
          />
          <label htmlFor="csv-file-input" className="btn-secondary" style={{ cursor: 'pointer' }}>
            <FileSpreadsheet size={18} /> {file ? file.name : 'Select CSV File'}
          </label>

          <button
            onClick={handleUpload}
            className="btn-primary"
            disabled={!file || loading}
          >
            {loading ? <RefreshCw size={18} className="animate-spin" /> : <UploadCloud size={18} />}
            {loading ? 'Processing Batch...' : 'Process Portfolio'}
          </button>
        </div>

        {error && (
          <p style={{ color: 'var(--status-danger)', fontSize: '0.88rem', marginTop: '16px' }}>
            {error}
          </p>
        )}
      </div>

      {/* Results Section */}
      {batchResult && (
        <div>
          {/* Summary Cards */}
          <div className="grid-4" style={{ marginBottom: '28px' }}>
            <div className="glass-card" style={{ padding: '20px' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', textTransform: 'uppercase' }}>Total Applications</span>
              <h3 style={{ fontSize: '1.8rem', fontWeight: 800, marginTop: '4px' }}>{batchResult.summary.total_applicants}</h3>
            </div>

            <div className="glass-card" style={{ padding: '20px', borderLeft: '4px solid var(--status-success)' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', textTransform: 'uppercase' }}>Approved Tiers</span>
              <h3 style={{ fontSize: '1.8rem', fontWeight: 800, marginTop: '4px', color: '#34d399' }}>{batchResult.summary.approved_count}</h3>
            </div>

            <div className="glass-card" style={{ padding: '20px', borderLeft: '4px solid var(--status-warning)' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', textTransform: 'uppercase' }}>Manual Review</span>
              <h3 style={{ fontSize: '1.8rem', fontWeight: 800, marginTop: '4px', color: '#fbbf24' }}>{batchResult.summary.review_count}</h3>
            </div>

            <div className="glass-card" style={{ padding: '20px', borderLeft: '4px solid var(--status-danger)' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', textTransform: 'uppercase' }}>Declined</span>
              <h3 style={{ fontSize: '1.8rem', fontWeight: 800, marginTop: '4px', color: '#f87171' }}>{batchResult.summary.declined_count}</h3>
            </div>
          </div>

          {/* Results Table Header & Export Button */}
          <div className="glass-card" style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 700 }}>Scored Applicant Portfolio</h3>
              <button onClick={handleExportCSV} className="btn-secondary">
                <Download size={16} /> Export CSV Report
              </button>
            </div>

            {/* Table */}
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.88rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-secondary)' }}>
                    <th style={{ padding: '12px 16px' }}>Applicant ID</th>
                    <th style={{ padding: '12px 16px' }}>Risk Score (%)</th>
                    <th style={{ padding: '12px 16px' }}>Probability</th>
                    <th style={{ padding: '12px 16px' }}>Category</th>
                    <th style={{ padding: '12px 16px' }}>Decision</th>
                    <th style={{ padding: '12px 16px' }}>Recommended Product Tier</th>
                  </tr>
                </thead>
                <tbody>
                  {batchResult.predictions.map((p, idx) => (
                    <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                      <td style={{ padding: '12px 16px', fontWeight: 600 }}>{p.applicant_id}</td>
                      <td style={{ padding: '12px 16px' }}>{p.risk_score}%</td>
                      <td style={{ padding: '12px 16px' }}>{p.default_probability}</td>
                      <td style={{ padding: '12px 16px' }}>{p.risk_category}</td>
                      <td style={{ padding: '12px 16px' }}>
                        <span className={
                          p.decision === 'APPROVED' ? 'badge-success' :
                          p.decision === 'MANUAL_REVIEW' ? 'badge-warning' : 'badge-danger'
                        } style={{ padding: '4px 10px', borderRadius: '8px', fontSize: '0.78rem', fontWeight: 600 }}>
                          {p.decision}
                        </span>
                      </td>
                      <td style={{ padding: '12px 16px', color: 'var(--text-secondary)' }}>{p.recommended_tier}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
