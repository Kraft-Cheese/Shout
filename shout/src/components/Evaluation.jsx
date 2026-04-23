import { useState, useRef } from 'preact/hooks';
import { State, isReady } from '../stores/state.js';
import { loadModel, transcribeBatch } from '../workers/worker.js';

const SAMPLE_RATE = 16000;

export function Evaluation() {
  const [csvFile, setCsvFile] = useState(null);
  const [audioFiles, setAudioFiles] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [results, setResults] = useState([]);

  const csvInputRef = useRef(null);
  const audioInputRef = useRef(null);

  const handleCsvChange = (e) => {
    setCsvFile(e.target.files[0]);
  };

  const handleAudioChange = (e) => {
    setAudioFiles(Array.from(e.target.files));
  };

  const parseCsv = async (file) => {
    const text = await file.text();
    const lines = text.split(/\r?\n/).filter(line => line.trim().length > 0);
    const delimiter = file.name.endsWith('.tsv') ? '\t' : (text.includes('\t') ? '\t' : ',');
    
    const headers = lines[0].split(delimiter).map(h => h.trim().toLowerCase());
    const data = lines.slice(1).map(line => {
      const values = line.split(delimiter);
      const row = {};
      headers.forEach((header, i) => {
        row[header] = values[i]?.trim();
      });
      return row;
    });
    return data;
  };

  const startEvaluation = async () => {
    if (!csvFile || audioFiles.length === 0) {
      State.error.value = "Please select both a CSV/TSV file and audio files.";
      return;
    }

    setIsProcessing(true);
    setResults([]);
    State.error.value = null;

    try {
      const csvData = await parseCsv(csvFile);
      // Expected CSV headers: file_name, transcription (or text/sentence), language (optional)
      const fileNameKey = Object.keys(csvData[0]).find(k => k.includes('file') || k.includes('id') || k.includes('path'));
      const textKey = Object.keys(csvData[0]).find(k => k.includes('transcription') || k.includes('text') || k.includes('sentence') || k.includes('label'));
      const langKey = Object.keys(csvData[0]).find(k => k.includes('lang'));

      if (!fileNameKey || !textKey) {
        throw new Error("CSV must contain 'file_name' and 'transcription' columns.");
      }

      const audioMap = new Map();
      audioFiles.forEach(f => audioMap.set(f.name, f));

      setProgress({ current: 0, total: csvData.length });

      const evaluationResults = [];

      for (let i = 0; i < csvData.length; i++) {
        const row = csvData[i];
        const fileName = row[fileNameKey];
        const referenceText = row[textKey];
        const targetLang = row[langKey] || State.language.value;

        const audioFile = audioMap.get(fileName) || audioFiles.find(f => f.name.includes(fileName));
        
        if (!audioFile) {
          console.warn(`Audio file not found for ${fileName}`);
          continue;
        }

        // Process audio
        const arrayBuffer = await audioFile.arrayBuffer();
        const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        const audio = audioBuffer.getChannelData(0);
        await audioCtx.close();

        // Ensure correct model is loaded if language specified in CSV
        if (targetLang !== State.language.value) {
            await loadModel(targetLang);
            State.language.value = targetLang;
        }

        const result = await transcribeBatch(audio, targetLang, referenceText);
        
        const baseResult = {
          file_name: fileName,
          language: targetLang,
          system: `whisper-${State.useQuantized.value ? 'q5' : 'full'}`,
          reference: referenceText,
          latency_ms: result.metrics.latency,
          rtf: result.metrics.rtf
        };

        evaluationResults.push({
          ...baseResult,
          raw_text: result.rawText,
          raw_wer: result.rawMetrics.wer,
          raw_cer: result.rawMetrics.cer,
          raw_f1_morpheme: result.rawMetrics.f1_morpheme,
          raw_f1_boundary: result.rawMetrics.f1_boundary,
          reconstruction: result.reconstruction,
          final_text: result.finalText,
          final_wer: result.finalMetrics.wer,
          final_cer: result.finalMetrics.cer,
          final_f1_morpheme: result.finalMetrics.f1_morpheme,
          final_f1_boundary: result.finalMetrics.f1_boundary,
        });

        setResults([...evaluationResults]);
        setProgress({ current: i + 1, total: csvData.length });
      }

      State.evalProgress.value = {
        ...State.evalProgress.value,
        status: 'done',
        results: evaluationResults
      };

    } catch (err) {
      State.error.value = `Evaluation failed: ${err.message}`;
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadResults = () => {
    if (results.length === 0) return;
    
    const headers = [
      'file_name', 'language', 'system', 'reference', 'latency_ms', 'rtf',
      'raw_text', 'raw_wer', 'raw_cer', 'raw_f1_morpheme', 'raw_f1_boundary',
      'reconstruction', 'final_text', 'final_wer', 'final_cer', 'final_f1_morpheme', 'final_f1_boundary'
    ];
    
    const generateCsv = (data) => [
      headers.join('\t'),
      ...data.map(r => headers.map(h => r[h]).join('\t'))
    ].join('\n');

    const download = (content, filename) => {
      const blob = new Blob([content], { type: 'text/tab-separated-values' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    };

    const timestamp = new Date().toISOString().slice(0,19).replace(/:/g,'-');
    download(generateCsv(results), `evaluation_results_${timestamp}.tsv`);
  };

  return (
    <div class="evaluation-section">
      <div class="section-label">Mass Evaluation (Batch Mode)</div>
      
      <div class="file-inputs">
        <div class="file-input-group">
          <label>CSV/TSV Metadata</label>
          <input 
            type="file" 
            accept=".csv,.tsv" 
            onChange={handleCsvChange}
            disabled={isProcessing}
          />
        </div>
        <div class="file-input-group">
          <label>Audio Files</label>
          <input 
            type="file" 
            accept="audio/*" 
            multiple 
            onChange={handleAudioChange}
            disabled={isProcessing}
          />
        </div>
      </div>

      <div class="eval-actions">
        <button 
          class="btn-primary" 
          onClick={startEvaluation}
          disabled={isProcessing || !csvFile || audioFiles.length === 0 || !isReady.value}
        >
          {isProcessing ? 'Processing...' : 'Start Evaluation'}
        </button>
        
        {results.length > 0 && !isProcessing && (
          <button class="btn-secondary" onClick={downloadResults}>
            Download TSV
          </button>
        )}
      </div>

      {isProcessing && (
        <div class="eval-progress-container">
          <div class="progress-info">
            <span>Processing {progress.current} / {progress.total}</span>
            <span>{Math.round((progress.current / progress.total) * 100)}%</span>
          </div>
          <div class="progress-bar">
            <div 
              class="progress-fill" 
              style={{ width: `${(progress.current / progress.total) * 100}%` }}
            />
          </div>
        </div>
      )}

      {results.length > 0 && (
        <div class="results-preview">
          <div class="section-label">Results Preview</div>
          <div class="results-table-container">
            <table>
              <thead>
                <tr>
                  <th>File</th>
                  <th>Recon.</th>
                  <th>Raw WER</th>
                  <th>Final WER</th>
                  <th>Final Output</th>
                </tr>
              </thead>
              <tbody>
                {results.slice(-5).reverse().map((r, i) => (
                  <tr key={i}>
                    <td>{r.file_name}</td>
                    <td>{r.reconstruction ? '✅' : '❌'}</td>
                    <td>{r.raw_wer.toFixed(3)}</td>
                    <td>{r.final_wer.toFixed(3)}</td>
                    <td>{r.final_text.slice(0, 50)}...</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
