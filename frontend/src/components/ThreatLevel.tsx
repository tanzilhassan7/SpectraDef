import React from 'react';

interface ThreatLevelProps {
  level?: 'LOW' | 'MEDIUM' | 'HIGH' | string;
}

export const ThreatLevel: React.FC<ThreatLevelProps> = ({ level = 'LOW' }) => {
  let bgColor = 'bg-green-800 text-green-200 border-green-600';
  if (level === 'MEDIUM') bgColor = 'bg-yellow-800 text-yellow-200 border-yellow-600';
  if (level === 'HIGH') bgColor = 'bg-red-800 text-red-200 border-red-600';

  return (
    <span className={`inline-block px-3 py-1 text-xs font-black tracking-wider rounded border ${bgColor}`}>
      THREAT LEVEL: {level}
    </span>
  );
};
