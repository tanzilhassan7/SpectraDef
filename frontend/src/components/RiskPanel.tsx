import React from 'react';
import { RiskEvaluation } from '../lib/api';
import { ThreatLevel } from './ThreatLevel';

interface RiskPanelProps {
  risk?: RiskEvaluation | null;
}

export const RiskPanel: React.FC<RiskPanelProps> = ({ risk }) => {
  if (!risk) {
    return (
      <div className="bg-gray-800 p-4 rounded border border-gray-700">
        <h3 className="font-bold text-gray-300">Risk Engine Analysis</h3>
        <p className="text-sm text-gray-500 italic mt-2">Shield OFF — Risk analysis bypassed</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 p-4 rounded border border-gray-700 space-y-3">
      <div className="flex justify-between items-center">
        <h3 className="font-bold text-gray-200">Risk Engine Analysis</h3>
        <ThreatLevel level={risk.risk_level} />
      </div>

      <div className="bg-gray-900 p-3 rounded border border-gray-700 flex items-center justify-between">
        <span className="text-sm text-gray-400 font-semibold">Anomalous Risk Score:</span>
        <span className="text-2xl font-black text-white">{risk.risk_score} / 100</span>
      </div>

      <div>
        <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">
          Triggered Risk Signals & Explanations:
        </h4>
        <ul className="list-disc list-inside text-xs text-gray-300 space-y-1">
          {risk.reasons.map((reason, idx) => (
            <li key={idx} className="bg-gray-900 px-2 py-1 rounded border border-gray-800">
              {reason}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};
