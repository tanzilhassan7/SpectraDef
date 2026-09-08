import React from 'react';

interface PipelineVisualizationProps {
  stages?: Record<string, string>;
}

export const PipelineVisualization: React.FC<PipelineVisualizationProps> = ({ stages }) => {
  if (!stages) return null;

  return (
    <div className="bg-gray-800 p-4 rounded border border-gray-700">
      <h3 className="font-bold text-gray-200 mb-2">ARGUS 7-Stage Pipeline Execution</h3>
      <div className="space-y-1 text-xs font-mono">
        {Object.entries(stages).map(([stage, status]) => (
          <div key={stage} className="flex justify-between bg-gray-900 px-3 py-1.5 rounded border border-gray-700">
            <span className="text-gray-300 font-bold uppercase">{stage.replace('_', ' ')}</span>
            <span
              className={
                status.includes('PASSED') || status.includes('COMPLETE') || status.includes('DELIVERED')
                  ? 'text-green-400 font-bold'
                  : status.includes('SKIPPED')
                  ? 'text-gray-500'
                  : 'text-yellow-400 font-bold'
              }
            >
              [{status}]
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};
