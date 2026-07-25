import React, { useEffect, useState } from 'react';
import { Navbar } from './components/Navbar';
import { SingleAssessment } from './components/SingleAssessment';
import { BatchUpload } from './components/BatchUpload';
import { DriftDashboard } from './components/DriftDashboard';
import { api } from './services/api';
import { ModelInfoResponse } from './types/credit';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'single' | 'batch' | 'drift'>('single');
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);

  useEffect(() => {
    api.getModelInfo().then(data => setModelInfo(data)).catch(err => console.error(err));
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} modelInfo={modelInfo} />

      <main style={{ flex: 1, maxWidth: '1280px', width: '100%', margin: '0 auto', padding: '0 24px 48px 24px' }}>
        {activeTab === 'single' && <SingleAssessment />}
        {activeTab === 'batch' && <BatchUpload />}
        {activeTab === 'drift' && <DriftDashboard />}
      </main>

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid var(--border-color)',
        padding: '24px 0',
        textAlign: 'center',
        color: 'var(--text-muted)',
        fontSize: '0.85rem'
      }}>
        <p>CreditRisk AI — Production Explainable Credit Decision Platform</p>
      </footer>
    </div>
  );
};

export default App;
