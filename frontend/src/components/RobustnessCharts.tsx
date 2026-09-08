import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { RobustnessSweepResponse } from '../lib/api';

interface RobustnessChartsProps {
  sweepData?: RobustnessSweepResponse | null;
}

export const RobustnessCharts: React.FC<RobustnessChartsProps> = ({ sweepData }) => {
  if (!sweepData || !sweepData.series || sweepData.series.length === 0) {
    return (
      <div className="bg-gray-800 p-4 rounded border border-gray-700">
        <h3 className="font-bold text-gray-300">Robustness Sweep Curves</h3>
        <p className="text-sm text-gray-500 italic mt-2">Run a robustness sweep to view perturbation stability charts.</p>
      </div>
    );
  }

  const formattedData = sweepData.series.map((point) => ({
    strength: point.strength,
    Confidence: Math.round(point.confidence * 100),
    Agreement: Math.round(point.agreement * 100),
    'Risk Score': point.risk_contribution,
  }));

  return (
    <div className="bg-gray-800 p-4 rounded border border-gray-700 space-y-4">
      <h3 className="font-bold text-gray-200">
        Robustness Sweep Curves (Transform: <span className="capitalize text-blue-400">{sweepData.transform_name}</span>)
      </h3>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={formattedData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="strength" stroke="#9CA3AF" label={{ value: 'Perturbation Strength', position: 'insideBottom', offset: -5 }} />
            <YAxis stroke="#9CA3AF" domain={[0, 100]} />
            <Tooltip contentStyle={{ backgroundColor: '#1F2937', borderColor: '#4B5563', color: '#F3F4F6' }} />
            <Legend />
            <Line type="monotone" dataKey="Confidence" stroke="#60A5FA" strokeWidth={2} />
            <Line type="monotone" dataKey="Agreement" stroke="#34D399" strokeWidth={2} />
            <Line type="monotone" dataKey="Risk Score" stroke="#F87171" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
