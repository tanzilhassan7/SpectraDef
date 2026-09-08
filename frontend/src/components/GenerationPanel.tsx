import React, { useState } from 'react';

interface GenerationPanelProps {
  onGenerate: (method: string, targetClass: number, strength: string, iterations: number) => void;
  loading: boolean;
}

export const GenerationPanel: React.FC<GenerationPanelProps> = ({ onGenerate, loading }) => {
  const [method, setMethod] = useState('iterative_target');
  const [targetClass, setTargetClass] = useState(9); // Ostrich
  const [strength, setStrength] = useState('moderate');
  const [iterations, setIterations] = useState(10);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onGenerate(method, targetClass, strength, iterations);
  };

  return (
    <form onSubmit={handleSubmit} className="bg-gray-800 p-4 rounded border border-gray-700 space-y-3">
      <h3 className="font-bold text-gray-200">Adversarial Generator Controls</h3>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 text-xs">
        <div>
          <label className="block text-gray-400 font-semibold mb-1">Attack Method</label>
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value)}
            className="w-full bg-gray-900 border border-gray-700 text-white rounded p-1.5"
          >
            <option value="iterative_target">Iterative Target Class (Default)</option>
            <option value="one_step_targeted">Targeted One-Step</option>
            <option value="iterative">Basic Iterative Method (Untargeted)</option>
            <option value="fgsm">FGSM (Untargeted)</option>
          </select>
        </div>

        <div>
          <label className="block text-gray-400 font-semibold mb-1">Target Class</label>
          <select
            value={targetClass}
            onChange={(e) => setTargetClass(Number(e.target.value))}
            className="w-full bg-gray-900 border border-gray-700 text-white rounded p-1.5"
          >
            <option value={9}>Ostrich (Class 9 - Benchmark Target)</option>
            <option value={292}>Lion (Class 292)</option>
            <option value={954}>Banana (Class 954)</option>
          </select>
        </div>

        <div>
          <label className="block text-gray-400 font-semibold mb-1">Epsilon Budget Preset</label>
          <select
            value={strength}
            onChange={(e) => setStrength(e.target.value)}
            className="w-full bg-gray-900 border border-gray-700 text-white rounded p-1.5"
          >
            <option value="controlled">Controlled (ε = 0.01)</option>
            <option value="moderate">Moderate (ε = 0.03)</option>
            <option value="strong">Strong (ε = 0.05)</option>
          </select>
        </div>

        <div>
          <label className="block text-gray-400 font-semibold mb-1">Iterations Preset</label>
          <select
            value={iterations}
            onChange={(e) => setIterations(Number(e.target.value))}
            className="w-full bg-gray-900 border border-gray-700 text-white rounded p-1.5"
          >
            <option value={5}>5 Iterations</option>
            <option value={10}>10 Iterations</option>
            <option value={20}>20 Iterations</option>
          </select>
        </div>
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 rounded text-sm disabled:opacity-50"
      >
        {loading ? 'Running Perturbation Generation & Live Model Verification...' : 'Generate Adversarial Image'}
      </button>
    </form>
  );
};
