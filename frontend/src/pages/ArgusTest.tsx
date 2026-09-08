import React, { useState, useEffect } from 'react';
import {
  analyzeArgus,
  fetchRobustnessSweep,
  ArgusAnalyzeResponse,
  RobustnessSweepResponse,
} from '../lib/api';
import { ShieldToggle } from '../components/ShieldToggle';
import { PredictionCard } from '../components/PredictionCard';
import { PipelineVisualization } from '../components/PipelineVisualization';
import { TransformationGrid } from '../components/TransformationGrid';
import { RiskPanel } from '../components/RiskPanel';
import { RobustnessCharts } from '../components/RobustnessCharts';

interface ArgusTestProps {
  initialArtifactId?: string | null;
}

export const ArgusTest: React.FC<ArgusTestProps> = ({ initialArtifactId }) => {
  const [shield, setShield] = useState<'on' | 'off'>('on');
  const [artifactId, setArtifactId] = useState<string | null>(initialArtifactId || null);
  const [file, setFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<ArgusAnalyzeResponse | null>(null);
  const [sweepData, setSweepData] = useState<RobustnessSweepResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (initialArtifactId) {
      setArtifactId(initialArtifactId);
      runAnalysis(initialArtifactId, shield);
    } else {
      runAnalysis(null, shield);
    }
  }, [initialArtifactId]);

  const runAnalysis = async (artId: string | null, shieldState: 'on' | 'off') => {
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('shield', shieldState);
      if (file) {
        formData.append('file', file);
      } else if (artId) {
        formData.append('artifact_id', artId);
      } else {
        formData.append('fixture_id', 'tiger_cat_benchmark');
      }

      const res = await analyzeArgus(formData);
      setAnalysis(res);

      if (shieldState === 'on') {
        const sweepForm = new FormData();
        sweepForm.append('transform_name', 'jpeg');
        if (file) sweepForm.append('file', file);
        else if (artId) sweepForm.append('artifact_id', artId);
        else sweepForm.append('fixture_id', 'tiger_cat_benchmark');
        
        const sweepRes = await fetchRobustnessSweep(sweepForm);
        setSweepData(sweepRes);
      } else {
        setSweepData(null);
      }
    } catch (err: any) {
      setError(err.message || 'ARGUS analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const handleShieldToggle = (newShield: 'on' | 'off') => {
    setShield(newShield);
    runAnalysis(artifactId, newShield);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setArtifactId(null);
      runAnalysis(null, shield);
    }
  };

  return (
    <div className="space-y-4">
      <div className="bg-gray-800 p-4 rounded border border-gray-700 flex flex-wrap justify-between items-center gap-4">
        <div>
          <h2 className="text-xl font-bold text-gray-100">ARGUS TEST — Robustness & Trust Verification</h2>
          <p className="text-xs text-gray-400">
            Verify prediction stability across controlled input transformations before trusting raw model output.
          </p>
        </div>

        <div className="flex items-center space-x-4">
          <ShieldToggle shield={shield} onToggle={handleShieldToggle} />
          <div>
            <label className="block text-xs font-semibold text-gray-400">Custom Upload:</label>
            <input type="file" accept="image/*" onChange={handleFileChange} className="text-xs text-gray-300" />
          </div>
        </div>
      </div>

      {artifactId && (
        <div className="bg-blue-950/50 border border-blue-700 text-blue-200 p-2 px-3 rounded text-xs">
          Loaded Adversarial Artifact: <span className="font-bold font-mono">{artifactId}</span>
        </div>
      )}

      {error && (
        <div className="bg-red-900/50 border border-red-700 text-red-200 p-3 rounded text-xs font-semibold">
          Error: {error}
        </div>
      )}

      {loading && (
        <div className="bg-gray-800 p-4 rounded text-center text-xs font-bold text-gray-400 animate-pulse">
          Running ARGUS Multi-Stage Robustness Pipeline Analysis...
        </div>
      )}

      {analysis && !loading && (
        <div className="space-y-4">
          {/* Top Row: Raw Prediction & Final Decision */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <PredictionCard title="Raw Base Model Prediction (InceptionV3)" prediction={analysis.raw_prediction} />

            <div className="bg-gray-800 p-4 rounded border border-gray-700 space-y-3">
              <h3 className="font-bold text-gray-200">ARGUS Mitigation & Final Decision</h3>
              <div className="bg-gray-900 p-3 rounded border border-gray-700">
                <div className="text-xs font-semibold text-gray-400 uppercase">Decision Status:</div>
                <div className="text-lg font-black text-white">{analysis.mitigation?.final_decision}</div>
              </div>

              <div className="bg-gray-900 p-3 rounded border border-gray-700">
                <div className="text-xs font-semibold text-gray-400 uppercase">Final Output Prediction:</div>
                <div className="text-2xl font-black text-green-400 capitalize">
                  {analysis.mitigation?.final_prediction}
                </div>
              </div>

              <div className="text-xs text-gray-300 bg-gray-900 p-2 rounded border border-gray-800">
                <span className="font-bold text-gray-400">Action Policy:</span> {analysis.mitigation?.mitigation_action}
              </div>
            </div>
          </div>

          {/* Pipeline Stages */}
          <PipelineVisualization stages={analysis.pipeline_stages} />

          {/* Transformations & Risk */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="md:col-span-2">
              <TransformationGrid views={analysis.transformed_views} />
            </div>
            <div className="md:col-span-1">
              <RiskPanel risk={analysis.risk} />
            </div>
          </div>

          {/* Consistency Metrics Panel */}
          {analysis.consistency && (
            <div className="bg-gray-800 p-4 rounded border border-gray-700">
              <h3 className="font-bold text-gray-200 mb-3">Consistency Engine Metrics</h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs text-center">
                <div className="bg-gray-900 p-2 rounded border border-gray-700">
                  <div className="text-gray-400 font-semibold">Prediction Agreement</div>
                  <div className="text-lg font-bold text-white">{(analysis.consistency.prediction_agreement * 100).toFixed(0)}%</div>
                </div>
                <div className="bg-gray-900 p-2 rounded border border-gray-700">
                  <div className="text-gray-400 font-semibold">Class Switching Rate</div>
                  <div className="text-lg font-bold text-white">{analysis.consistency.prediction_switching_rate} changes</div>
                </div>
                <div className="bg-gray-900 p-2 rounded border border-gray-700">
                  <div className="text-gray-400 font-semibold">Dominant Consensus</div>
                  <div className="text-lg font-bold text-blue-400 capitalize">{analysis.consistency.dominant_prediction}</div>
                </div>
                <div className="bg-gray-900 p-2 rounded border border-gray-700">
                  <div className="text-gray-400 font-semibold">Mean Conf Std-Dev</div>
                  <div className="text-lg font-bold text-white">&plusmn;{analysis.consistency.std_confidence.toFixed(3)}</div>
                </div>
              </div>
            </div>
          )}

          {/* Robustness Sweep Charts */}
          <RobustnessCharts sweepData={sweepData} />
        </div>
      )}
    </div>
  );
};
