import { useState } from 'react';
import './App.css';
import { UploadView } from './components/UploadView/UploadView';

function App() {
  const [currentView, setCurrentView] = useState<'upload' | 'results'>('upload');
  const [fileName, setFileName] = useState<string>('');

  const handleAnalyze = (name: string) => {
    setFileName(name);
    setCurrentView('results');
  };

  return (
    <>
      {currentView === 'upload' ? (
        <UploadView onAnalyze={handleAnalyze} />
      ) : (
        <div>Results View Placeholder (File: {fileName})</div>
      )}
    </>
  );
}

export default App;
