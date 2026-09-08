import React, { useState, useEffect } from 'react';
import {
  getFixtures,
  generateAdversarial,
  predictImage,
  FixtureItem,
  AdversarialGenerateResponse,
  PredictionResponse,
} from '../lib/api';
import { PredictionCard } from '../components/PredictionCard';
import { GenerationPanel } from '../components/GenerationPanel';
import { ImageComparison } from '../components/ImageComparison';

interface AdversarialLabProps {
  onSendToArgus: (artifactId: string) => void;
}

export const AdversarialLab: React.FC<AdversarialLabProps> = ({ onSendToArgus }) => {
  const [fixtures, setFixtures] = useState<FixtureItem[]>([]);
  const [selectedFixture, setSelectedFixture] = useState<string>('tiger_cat_benchmark');
  const [file, setFile] = useState<File | null>(null);
  const [cleanPreviewUrl, setCleanPreviewUrl] = useState<string | null>(null);
  const [cleanPrediction, setCleanPrediction] = useState<PredictionResponse | null>(null);
  const [generationResult, setGenerationResult] = useState<AdversarialGenerateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getFixtures()
      .then((items) => {
        setFixtures(items);
        if (items.length > 0) {
          setSelectedFixture(items[0].id);
          loadCleanPredictionForFixture(items[0].id);
        }
      })
      .catch((err) => console.error(err));
  }, []);

  const loadCleanPredictionForFixture = async (fixtureId: string) => {
    try {
      setFile(null);
      setCleanPreviewUrl(`/api/fixtures/${fixtureId}`);
      const formData = new FormData();
      formData.append('fixture_id', fixtureId);
      const res = await predictImage(formData);
      setCleanPrediction(res);
      setGenerationResult(null);
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setCleanPreviewUrl(URL.createObjectURL(selectedFile));
      setSelectedFixture('');
      
      try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        const res = await predictImage(formData);
        setCleanPrediction(res);
        setGenerationResult(null);
      } catch (err: any) {
        setError(err.message);
      }
    }
  };

  const handleGenerate = async (
    method: string,
    targetClass: number,
    strengthPreset: string,
    iterationsPreset: number
  ) => {
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('method', method);
      formData.append('target_class', targetClass.toString());
      formData.append('strength_preset', strengthPreset);
      formData.append('iterations_preset', iterationsPreset.toString());

      if (file) {
        formData.append('file', file);
      } else if (selectedFixture) {
        formData.append('fixture_id', selectedFixture);
      }

      const res = await generateAdversarial(formData);
      setGenerationResult(res);
    } catch (err: any) {
      setError(err.message || 'Generation failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="bg-gray-800 p-4 rounded border border-gray-700 space-y-3">
        <h2 className="text-xl font-bold text-gray-100">Adversarial Lab — Controlled Input Perturbation</h2>
        <p className="text-xs text-gray-400">
          Generate bounded adversarial perturbations locally against InceptionV3. Test model stability and verify raw target output.
        </p>

        <div className="flex flex-wrap gap-4 items-center text-xs">
          <div>
            <label className="font-semibold text-gray-300 mr-2">Benchmark Fixture:</label>
            <select
              value={selectedFixture}
              onChange={(e) => {
                setSelectedFixture(e.target.value);
                loadCleanPredictionForFixture(e.target.value);
              }}
              className="bg-gray-900 border border-gray-700 text-white rounded p-1.5"
            >
              {fixtures.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.name} ({f.ground_truth_label})
                </option>
              ))}
            </select>
          </div>

          <div className="text-gray-500">OR</div>

          <div>
            <label className="font-semibold text-gray-300 mr-2">Upload Image:</label>
            <input type="file" accept="image/*" onChange={handleFileChange} className="text-gray-300 text-xs" />
          </div>
        </div>
      </div>

      {error && (
        <div className="bg-red-900/50 border border-red-700 text-red-200 p-3 rounded text-xs font-semibold">
          Error: {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="md:col-span-1">
          <PredictionCard title="Clean Base Model Prediction" prediction={cleanPrediction} />
        </div>
        <div className="md:col-span-2">
          <GenerationPanel onGenerate={handleGenerate} loading={loading} />
        </div>
      </div>

      {generationResult && (
        <div className="space-y-4">
          {generationResult.generation_status === 'failed' && (
            <div className="bg-yellow-900/50 border border-yellow-600 text-yellow-200 p-3 rounded text-xs font-bold">
              Generation Status: FAILED — {generationResult.status_reason}
            </div>
          )}

          <ImageComparison
            cleanImageUrl={cleanPreviewUrl}
            cleanPrediction={cleanPrediction?.top1_label}
            cleanConfidence={cleanPrediction?.top1_confidence}
            adversarialImageUrl={`/api/adversarial/artifact/${generationResult.id}`}
            adversarialPrediction={generationResult.adversarial_prediction}
            adversarialConfidence={generationResult.adversarial_confidence}
            diffImageBase64={generationResult.diff_image_base64}
            artifactId={generationResult.id}
          />

          <div className="bg-gray-800 p-4 rounded border border-gray-700 flex justify-between items-center">
            <div className="text-xs text-gray-300">
              <span className="font-bold text-gray-100">Artifact ID:</span> {generationResult.id} |{' '}
              <span className="font-bold text-gray-100">Status:</span> {generationResult.generation_status} |{' '}
              <span className="font-bold text-gray-100">Transition:</span> {generationResult.source_prediction} &rarr;{' '}
              <span className="text-red-400 font-bold capitalize">{generationResult.adversarial_prediction}</span>
            </div>

            <div className="flex items-center space-x-2">
              <a
                href={`/api/adversarial/${generationResult.id}/download`}
                download={`adversarial_${generationResult.id}.png`}
                className="bg-blue-600 hover:bg-blue-700 text-white font-bold px-3 py-2 rounded text-xs"
              >
                &darr; Download PNG
              </a>

              <button
                onClick={() => onSendToArgus(generationResult.id)}
                className="bg-green-600 hover:bg-green-700 text-white font-bold px-4 py-2 rounded text-xs"
              >
                Send to ARGUS Test &rarr;
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
