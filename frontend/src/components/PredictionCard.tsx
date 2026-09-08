import React from 'react';
import { PredictionResponse } from '../lib/api';

interface PredictionCardProps {
  title?: string;
  prediction?: PredictionResponse | null;
}

export const PredictionCard: React.FC<PredictionCardProps> = ({
  title = "Model Prediction",
  prediction,
}) => {
  if (!prediction) {
    return (
      <div className="bg-gray-800 p-4 rounded border border-gray-700">
        <h3 className="font-bold text-gray-300">{title}</h3>
        <p className="text-sm text-gray-500 italic mt-2">No prediction available</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 p-4 rounded border border-gray-700">
      <h3 className="font-bold text-gray-300 mb-2">{title}</h3>
      <div className="mb-3">
        <div className="text-2xl font-extrabold text-blue-400 capitalize">
          {prediction.top1_label}
        </div>
        <div className="text-sm text-gray-400">
          Confidence: <span className="font-bold text-white">{(prediction.top1_confidence * 100).toFixed(1)}%</span>
          <span className="ml-3 text-xs text-gray-500">Latency: {prediction.latency_ms}ms</span>
        </div>
      </div>

      <div className="text-xs text-gray-400 mt-2">
        <div className="font-semibold mb-1">Top-5 Probabilities:</div>
        <div className="space-y-1">
          {prediction.top5.map((item, idx) => (
            <div key={idx} className="flex justify-between border-b border-gray-700 py-0.5">
              <span className="capitalize">{item.label}</span>
              <span className="font-mono font-semibold">{(item.confidence * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
