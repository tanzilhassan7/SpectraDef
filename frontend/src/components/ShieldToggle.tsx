import React from 'react';

interface ShieldToggleProps {
  shield: 'on' | 'off';
  onToggle: (newShield: 'on' | 'off') => void;
}

export const ShieldToggle: React.FC<ShieldToggleProps> = ({ shield, onToggle }) => {
  return (
    <div className="flex items-center space-x-3 bg-gray-800 p-2 rounded border border-gray-700">
      <span className="font-semibold text-sm">ARGUS SHIELD:</span>
      <button
        onClick={() => onToggle('off')}
        className={`px-3 py-1 text-xs font-bold rounded ${
          shield === 'off' ? 'bg-red-600 text-white' : 'bg-gray-700 text-gray-400'
        }`}
      >
        OFF (RAW INFERENCE)
      </button>
      <button
        onClick={() => onToggle('on')}
        className={`px-3 py-1 text-xs font-bold rounded ${
          shield === 'on' ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-400'
        }`}
      >
        ON (ROBUSTNESS LAYER)
      </button>
    </div>
  );
};
