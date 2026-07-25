import React, { useEffect, useState } from 'react';
import { api } from '../services/api';
import { DriftReportResponse } from '../types/credit';
import { Activity, AlertTriangle, CheckCircle2, RefreshCw, BarChart2, ShieldCheck } from 'lucide-react';

export const DriftDashboard: React.FC = () => {
  const [driftReport, setDriftReport] = useState<DriftReportResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const fetchDrift = async () => {
    setLoading(true);
    try {
      const data = await api.getDriftReport();
      setDriftReport(data);
    } catch (err) {
      console.error('Failed to load drift report:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDrift();
  }, []);

  return (
    <div style={{ padding: '32px 0' }}>
      {/* Header & Refresh Button */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '28px', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h2 style={{ fontSize: '1.4rem', fontWeight: 800 }}>Model Health & Data Drift Telemetry</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem' }}>
            Real-time statistical drift monitoring comparing baseline training distribution against production stream.
          </p>
        </div>

        <button onClick={fetchDrift} className="btn-secondary">
          <RefreshCw size={16} className={loading ? 'animate-spin' : ''} /> Refresh Telemetry
        </button>
      </div>

      {loading ? (
        <div className="glass-card" style={{ padding: '48px', textAlign: 'center', color: 'var(--text-secondary)' }}>
          <Activity size={32} className="animate-spin" style={{ margin: '0 auto 16px auto', color: 'var(--accent-cyan)' }} />
          <p>Calculating Kolmogorov-Smirnov statistical feature drift metrics...</p>
        </div>
      ) : driftReport ? (
        <div>
          {/* Top Status Cards */}
          <div className="grid-3" style={{ marginBottom: '28px' }}>
            <div className="glass-card" style={{ padding: '24px', borderLeft: `6px solid ${driftReport.dataset_drift ? 'var(--status-danger)' : 'var(--status-success)'}` }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', textTransform: 'uppercase' }}>Dataset Drift Status</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '8px' }}>
                {driftReport.dataset_drift ? (
                  <>
                    <AlertTriangle size={28} color="var(--status-danger)" />
                    <div>
                      <h3 style={{ fontSize: '1.3rem', fontWeight: 800, color: 'var(--status-danger)' }}>Drift Detected</h3>
                      <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>Retrain pipeline recommended</p>
                    </div>
                  </>
                ) : (
                  <>
                    <CheckCircle2 size={28} color="var(--status-success)" />
                    <div>
                      <h3 style={{ fontSize: '1.3rem', fontWeight: 800, color: 'var(--status-success)' }}>Stable Distribution</h3>
                      <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>No retrain trigger needed</p>
                    </div>
                  </>
                )}
              </div>
            </div>

            <div className="glass-card" style={{ padding: '24px' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', textTransform: 'uppercase' }}>Drifted Feature Share</span>
              <h3 style={{ fontSize: '1.8rem', fontWeight: 800, marginTop: '4px', color: 'var(--accent-cyan)' }}>
                {(driftReport.drift_share * 100).toFixed(1)}%
              </h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                {driftReport.number_of_drifted_columns} of {driftReport.total_columns} features shifted
              </p>
            </div>

            <div className="glass-card" style={{ padding: '24px' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', textTransform: 'uppercase' }}>Statistical Method</span>
              <h3 style={{ fontSize: '1.3rem', fontWeight: 800, marginTop: '4px', color: 'var(--text-primary)' }}>
                Kolmogorov-Smirnov
              </h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                p-value threshold = 0.05
              </p>
            </div>
          </div>

          {/* Feature Drift Table */}
          <div className="glass-card" style={{ padding: '24px' }}>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '20px' }}>
              Feature-Level Statistical Breakdown
            </h3>

            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.88rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-secondary)' }}>
                    <th style={{ padding: '12px 16px' }}>Feature Name</th>
                    <th style={{ padding: '12px 16px' }}>Status</th>
                    <th style={{ padding: '12px 16px' }}>KS Statistic</th>
                    <th style={{ padding: '12px 16px' }}>p-Value</th>
                    <th style={{ padding: '12px 16px' }}>Threshold</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(driftReport.drift_by_columns).map(([col, detail], idx) => (
                    <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                      <td style={{ padding: '12px 16px', fontWeight: 600 }}>{col}</td>
                      <td style={{ padding: '12px 16px' }}>
                        <span className={detail.drift_detected ? 'badge-danger' : 'badge-success'} style={{ padding: '4px 10px', borderRadius: '8px', fontSize: '0.78rem', fontWeight: 600 }}>
                          {detail.drift_detected ? 'DRIFTED' : 'STABLE'}
                        </span>
                      </td>
                      <td style={{ padding: '12px 16px' }}>{detail.ks_statistic}</td>
                      <td style={{ padding: '12px 16px' }}>{detail.p_value}</td>
                      <td style={{ padding: '12px 16px', color: 'var(--text-secondary)' }}>{detail.threshold}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};
