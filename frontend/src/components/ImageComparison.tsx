import React from 'react';

interface ImageComparisonProps {
  cleanImageUrl?: string | null;
  cleanPrediction?: string;
  cleanConfidence?: number;
  adversarialImageUrl?: string | null;
  adversarialPrediction?: string;
  adversarialConfidence?: number;
  diffImageBase64?: string | null;
  artifactId?: string | null;
}

export const ImageComparison: React.FC<ImageComparisonProps> = ({
  cleanImageUrl,
  cleanPrediction,
  cleanConfidence,
  adversarialImageUrl,
  adversarialPrediction,
  adversarialConfidence,
  diffImageBase64,
  artifactId,
}) => {
  return (
    <div className="bg-gray-800 p-4 rounded border border-gray-700">
      <div className="flex justify-between items-center mb-3">
        <h3 className="font-bold text-gray-200">Clean vs Adversarial Image Comparison</h3>
        {artifactId && (
          <a
            href={`/api/adversarial/${artifactId}/download`}
            download={`adversarial_${artifactId}.png`}
            className="bg-blue-600 hover:bg-blue-700 text-white font-bold px-3 py-1.5 rounded text-xs inline-flex items-center space-x-1"
          >
            <span>&darr; Download Adversarial PNG</span>
          </a>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
        <div className="bg-gray-900 p-3 rounded border border-gray-700">
          <div className="font-semibold text-sm text-gray-300 mb-2">Original Clean Input</div>
          {cleanImageUrl ? (
            <img src={cleanImageUrl} alt="Clean" className="w-full h-48 object-contain rounded mb-2 border border-gray-800" />
          ) : (
            <div className="w-full h-48 bg-gray-800 rounded mb-2 flex items-center justify-center text-xs text-gray-500">No Image</div>
          )}
          <div className="text-sm font-bold text-blue-400 capitalize">{cleanPrediction || '—'}</div>
          <div className="text-xs text-gray-400">{cleanConfidence ? `${(cleanConfidence * 100).toFixed(1)}%` : ''}</div>
        </div>

        <div className="bg-gray-900 p-3 rounded border border-gray-700">
          <div className="font-semibold text-sm text-gray-300 mb-2">Amplified Difference Visualization</div>
          {diffImageBase64 ? (
            <img src={diffImageBase64} alt="Difference" className="w-full h-48 object-contain rounded mb-2 border border-gray-800" />
          ) : (
            <div className="w-full h-48 bg-gray-800 rounded mb-2 flex items-center justify-center text-xs text-gray-500">No Difference Data</div>
          )}
          <div className="text-xs text-yellow-400 font-semibold">Amplified 10x for visual clarity</div>
        </div>

        <div className="bg-gray-900 p-3 rounded border border-gray-700">
          <div className="font-semibold text-sm text-gray-300 mb-2">Generated Adversarial Image</div>
          {adversarialImageUrl ? (
            <img src={adversarialImageUrl} alt="Adversarial" className="w-full h-48 object-contain rounded mb-2 border border-gray-800" />
          ) : (
            <div className="w-full h-48 bg-gray-800 rounded mb-2 flex items-center justify-center text-xs text-gray-500">No Image</div>
          )}
          <div className="text-sm font-bold text-red-400 capitalize">{adversarialPrediction || '—'}</div>
          <div className="text-xs text-gray-400">{adversarialConfidence ? `${(adversarialConfidence * 100).toFixed(1)}%` : ''}</div>
        </div>
      </div>
    </div>
  );
};
