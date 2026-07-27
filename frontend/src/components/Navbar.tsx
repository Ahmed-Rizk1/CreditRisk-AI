import React, { useState } from 'react';
import { ShieldCheck, Cpu, Activity, UploadCloud, BarChart3, Settings, Link, Check, X } from 'lucide-react';
import { ModelInfoResponse } from '../types/credit';
import { getEffectiveApiBaseUrl } from '../services/api';

interface NavbarProps {
  activeTab: 'single' | 'batch' | 'drift';
  setActiveTab: (tab: 'single' | 'batch' | 'drift') => void;
  modelInfo: ModelInfoResponse | null;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, modelInfo }) => {
  const [showConfigModal, setShowConfigModal] = useState<boolean>(false);
  const [customUrl, setCustomUrl] = useState<string>(
    localStorage.getItem('CREDITRISK_API_URL') || getEffectiveApiBaseUrl()
  );
  const [savedSuccess, setSavedSuccess] = useState<boolean>(false);

  const handleSaveUrl = (e: React.FormEvent) => {
    e.preventDefault();
    if (customUrl.trim()) {
      localStorage.setItem('CREDITRISK_API_URL', customUrl.trim());
    } else {
      localStorage.removeItem('CREDITRISK_API_URL');
    }
    setSavedSuccess(true);
    setTimeout(() => {
      setSavedSuccess(false);
      setShowConfigModal(false);
      window.location.reload();
    }, 800);
  };

  const handleResetUrl = () => {
    localStorage.removeItem('CREDITRISK_API_URL');
    setCustomUrl(getEffectiveApiBaseUrl());
    window.location.reload();
  };

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

        {/* Right Action Bar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
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

          {/* Config Settings Modal Trigger */}
          <button
            onClick={() => setShowConfigModal(true)}
            className="btn-secondary"
            title="Configure Backend Target API URL"
            style={{ padding: '8px', borderRadius: '50%' }}
          >
            <Settings size={18} />
          </button>
        </div>
      </div>

      {/* Backend API Configuration Modal */}
      {showConfigModal && (
        <div style={{
          position: 'fixed',
          top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(0, 0, 0, 0.75)',
          backdropFilter: 'blur(4px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 100
        }}>
          <div className="glass-card" style={{ width: '100%', maxWidth: '520px', padding: '28px', margin: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Link size={20} color="var(--accent-cyan)" /> API Backend Target URL
              </h3>
              <button onClick={() => setShowConfigModal(false)} style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer' }}>
                <X size={20} />
              </button>
            </div>

            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
              Current target URL: <code style={{ color: 'var(--accent-cyan)', background: 'rgba(15,23,42,0.8)', padding: '2px 6px', borderRadius: '4px' }}>{getEffectiveApiBaseUrl()}</code>
            </p>

            <form onSubmit={handleSaveUrl}>
              <label className="input-label">Custom Render / Backend API Base URL</label>
              <input
                type="text"
                value={customUrl}
                onChange={(e) => setCustomUrl(e.target.value)}
                placeholder="https://creditrisk-backend.onrender.com/api/v1"
                className="input-field"
                style={{ marginBottom: '16px' }}
              />

              <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
                <button type="button" onClick={handleResetUrl} className="btn-secondary" style={{ fontSize: '0.85rem' }}>
                  Reset Default
                </button>
                <button type="submit" className="btn-primary" style={{ fontSize: '0.85rem' }}>
                  {savedSuccess ? <><Check size={16} /> Saved!</> : 'Save & Reload'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </header>
  );
};

