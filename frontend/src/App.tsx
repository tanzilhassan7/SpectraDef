import React, { useState, useEffect } from 'react';
import { AdversarialLab } from './pages/AdversarialLab';
import { ArgusTest } from './pages/ArgusTest';
import { runScriptedDemo } from './lib/api';

export default function App() {
  const [activeTab, setActiveTab] = useState<'lab' | 'argus'>('lab');
  const [selectedArtifactId, setSelectedArtifactId] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<{ status: string; device: string } | null>(null);
  const [demoLoading, setDemoLoading] = useState(false);

  useEffect(() => {
    fetch('/api/')
      .then((res) => res.json())
      .then((data) => setBackendStatus(data))
      .catch((err) => console.error('Backend status check failed:', err));
  }, []);

  const handleSendToArgus = (artifactId: string) => {
    setSelectedArtifactId(artifactId);
    setActiveTab('argus');
  };

  const handleRunScriptedDemo = async () => {
    setDemoLoading(true);
    try {
      const demoRes = await runScriptedDemo();
      if (demoRes.adversarial_generation?.id) {
        setSelectedArtifactId(demoRes.adversarial_generation.id);
        setActiveTab('argus');
      }
    } catch (e) {
      console.error('Demo script error:', e);
      alert('Demo script failed: ' + e);
    } finally {
      setDemoLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-4 space-y-4">
      {/* Header */}
      <header className="bg-gray-800 p-4 rounded border border-gray-700 flex flex-wrap justify-between items-center gap-4">
        <div>
          <h1 className="text-2xl font-black tracking-wider text-blue-400">ARGUS SHIELD</h1>
          <p className="text-xs text-gray-400">
            Don't just trust what your model sees. Verify that it deserves to be trusted.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="bg-gray-900 px-2.5 py-1 rounded border border-gray-700 font-mono text-gray-300">
            Model: InceptionV3
          </span>
          <span className="bg-gray-900 px-2.5 py-1 rounded border border-gray-700 font-mono text-gray-300">
            Dataset: ImageNet
          </span>
          <span className="bg-gray-900 px-2.5 py-1 rounded border border-gray-700 font-mono text-green-400">
            Device: {backendStatus?.device ? backendStatus.device.toUpperCase() : 'CPU'}
          </span>

          <button
            onClick={handleRunScriptedDemo}
            disabled={demoLoading}
            className="bg-purple-700 hover:bg-purple-800 text-white font-bold px-3 py-1 rounded border border-purple-600 disabled:opacity-50"
          >
            {demoLoading ? 'Executing Demo...' : 'Run Automated Demo Script'}
          </button>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="flex space-x-2 border-b border-gray-700 pb-2">
        <button
          onClick={() => setActiveTab('lab')}
          className={`px-4 py-2 text-xs font-bold rounded ${
            activeTab === 'lab' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          Tab 1: Adversarial Lab
        </button>
        <button
          onClick={() => setActiveTab('argus')}
          className={`px-4 py-2 text-xs font-bold rounded ${
            activeTab === 'argus' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          Tab 2: ARGUS Test
        </button>
      </div>

      {/* Main Content */}
      <main>
        {activeTab === 'lab' && <AdversarialLab onSendToArgus={handleSendToArgus} />}
        {activeTab === 'argus' && <ArgusTest initialArtifactId={selectedArtifactId} />}
      </main>

      {/* Footer */}
      <footer className="text-center text-xs text-gray-500 py-4 border-t border-gray-800">
        ARGUS SHIELD &copy; 2026 — Defensive Vision Robustness & Trust Benchmark
      </footer>
    </div>
  );
}
