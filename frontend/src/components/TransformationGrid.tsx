import React from 'react';
import { TransformedViewItem } from '../lib/api';

interface TransformationGridProps {
  views?: TransformedViewItem[] | null;
}

export const TransformationGrid: React.FC<TransformationGridProps> = ({ views }) => {
  if (!views || views.length === 0) {
    return (
      <div className="bg-gray-800 p-4 rounded border border-gray-700">
        <h3 className="font-bold text-gray-300">Multi-View Transformations</h3>
        <p className="text-sm text-gray-500 italic mt-2">Shield OFF — Multi-view transformation skipped</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 p-4 rounded border border-gray-700">
      <h3 className="font-bold text-gray-200 mb-3">Multi-View Transformation Inference ({views.length} Views)</h3>
      <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-2">
        {views.map((v, idx) => (
          <div key={idx} className="bg-gray-900 p-2 rounded border border-gray-700 text-center">
            {v.image_base64 ? (
              <img
                src={v.image_base64}
                alt={v.name}
                className="w-full h-20 object-cover rounded mb-1 border border-gray-800"
              />
            ) : (
              <div className="w-full h-20 bg-gray-800 rounded mb-1 flex items-center justify-center text-xs text-gray-500">
                No Img
              </div>
            )}
            <div className="text-xs font-bold capitalize text-blue-300">{v.name}</div>
            <div className="text-xs font-semibold text-gray-200 truncate capitalize">{v.prediction}</div>
            <div className="text-xs text-gray-400">{(v.confidence * 100).toFixed(0)}%</div>
          </div>
        ))}
      </div>
    </div>
  );
};
