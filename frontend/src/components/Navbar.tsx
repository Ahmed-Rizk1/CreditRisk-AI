import React from 'react';
import { ShieldCheck, Cpu, Activity, UploadCloud, BarChart3 } from 'lucide-react';
import { ModelInfoResponse } from '../types/credit';

interface NavbarProps {
  activeTab: 'single' | 'batch' | 'drift';
  setActiveTab: (tab: 'single' | 'batch' | 'drift') => void;
  modelInfo: ModelInfoResponse | null;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, modelInfo }) => {
  return (
    <header style={{
      borderBottom: '1px solid var(--border-color)',
      background: 'rgba(9, 13, 22, 0.85)',
      backdropFilter: 'blur(12px)',
      position: 'sticky',
      top: 0,
      zIndex: 50
    }}>
      <div style={{
        maxWidth: '1280px',
        margin: '0 auto',
        padding: '16px 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '16px'
      }}>
        {/* Brand Logo & Title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '42px',
            height: '42px',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, var(--accent-indigo), var(--accent-cyan))',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 16px rgba(99, 102, 241, 0.4)'
          }}>
            <ShieldCheck size={26} color="#ffffff" />
          </div>
          <div>
            <h1 style={{ fontSize: '1.35rem', fontWeight: 800, color: 'var(--text-primary)' }}>
              CreditRisk <span style={{ color: 'var(--accent-cyan)' }}>AI</span>
            </h1>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
              Explainable Decision & Telemetry Platform
            </p>
          </div>
        </div>

        {/* Tab Navigation */}
        <div style={{
          display: 'flex',
          gap: '8px',
          background: 'rgba(15, 23, 42, 0.8)',
          padding: '4px',
          borderRadius: '12px',
          border: '1px solid var(--border-color)'
        }}>
          <button
            onClick={() => setActiveTab('single')}
            className={activeTab === 'single' ? 'btn-primary' : 'btn-secondary'}
            style={{ padding: '8px 16px', fontSize: '0.88rem' }}
          >
            <Activity size={16} /> Single Assessment
          </button>

          <button
            onClick={() => setActiveTab('batch')}
            className={activeTab === 'batch' ? 'btn-primary' : 'btn-secondary'}
            style={{ padding: '8px 16px', fontSize: '0.88rem' }}
          >
            <UploadCloud size={16} /> Batch CSV Scoring
          </button>

          <button
            onClick={() => setActiveTab('drift')}
            className={activeTab === 'drift' ? 'btn-primary' : 'btn-secondary'}
            style={{ padding: '8px 16px', fontSize: '0.88rem' }}
          >
            <BarChart3 size={16} /> Model Health & Drift
          </button>
        </div>

        {/* Model Status Badge */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          background: 'rgba(30, 41, 59, 0.6)',
          padding: '6px 14px',
          borderRadius: '20px',
          border: '1px solid var(--border-color)',
          fontSize: '0.82rem'
        }}>
          <Cpu size={16} color="var(--accent-cyan)" />
          <span style={{ color: 'var(--text-secondary)' }}>
            Model: <strong style={{ color: 'var(--text-primary)' }}>{modelInfo?.model_name || 'CatBoost'}</strong>
          </span>
          <span className="badge-success" style={{ padding: '2px 8px', borderRadius: '10px', fontSize: '0.72rem', fontWeight: 600 }}>
            {modelInfo?.stage || 'Production'}
          </span>
        </div>
      </div>
    </header>
  );
};
