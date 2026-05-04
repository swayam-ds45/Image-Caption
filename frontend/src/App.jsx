import React, { useState, useRef } from 'react';
import axios from 'axios';
import { UploadCloud, Image as ImageIcon, Loader2, Sparkles } from 'lucide-react';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [caption, setCaption] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  // In production, this should be your deployed Render backend URL
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (!selectedFile.type.startsWith('image/')) {
        setError('Please upload a valid image file.');
        return;
      }
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setCaption('');
      setError('');
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      if (!droppedFile.type.startsWith('image/')) {
        setError('Please upload a valid image file.');
        return;
      }
      setFile(droppedFile);
      setPreview(URL.createObjectURL(droppedFile));
      setCaption('');
      setError('');
    }
  };

  const generateCaption = async () => {
    if (!file) return;

    setLoading(true);
    setError('');
    
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post(`${API_URL}/api/generate-caption`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      if (response.data.success) {
        setCaption(response.data.caption);
      } else {
        setError(response.data.message || 'Failed to generate caption.');
      }
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.message || 'Server error. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col items-center py-12 px-4 sm:px-6 lg:px-8 font-sans">
      
      <div className="w-full max-w-3xl space-y-8">
        <div className="text-center">
          <h1 className="text-4xl font-extrabold text-slate-900 tracking-tight sm:text-5xl flex items-center justify-center gap-3">
            <Sparkles className="w-10 h-10 text-indigo-600" />
            AI Image Captioner
          </h1>
          <p className="mt-4 text-lg text-slate-600">
            Upload any image and let our deep learning model describe it for you.
          </p>
        </div>

        <div className="bg-white p-8 rounded-3xl shadow-xl border border-slate-100">
          
          {!preview ? (
            <div 
              className="border-4 border-dashed border-slate-200 rounded-2xl p-12 flex flex-col items-center justify-center cursor-pointer hover:bg-slate-50 hover:border-indigo-400 transition-colors duration-300"
              onClick={() => fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              <UploadCloud className="w-16 h-16 text-slate-400 mb-4" />
              <p className="text-xl font-medium text-slate-700">Drag & drop your image here</p>
              <p className="text-sm text-slate-500 mt-2">or click to browse from your computer</p>
              <input 
                type="file" 
                className="hidden" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                accept="image/*"
              />
            </div>
          ) : (
            <div className="space-y-6">
              <div className="relative group rounded-2xl overflow-hidden bg-slate-100 border border-slate-200">
                <img 
                  src={preview} 
                  alt="Preview" 
                  className="w-full max-h-[400px] object-contain"
                />
                <button 
                  onClick={() => {
                    setFile(null);
                    setPreview(null);
                    setCaption('');
                  }}
                  className="absolute top-4 right-4 bg-white/90 backdrop-blur text-slate-700 px-4 py-2 rounded-full text-sm font-semibold shadow-sm hover:bg-white transition-colors"
                >
                  Change Image
                </button>
              </div>

              {!caption && !loading && (
                <button 
                  onClick={generateCaption}
                  className="w-full py-4 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl font-bold text-lg shadow-lg hover:shadow-indigo-500/30 transition-all active:scale-[0.98] flex items-center justify-center gap-2"
                >
                  <Sparkles className="w-5 h-5" />
                  Generate Caption
                </button>
              )}

              {loading && (
                <div className="w-full py-8 flex flex-col items-center justify-center space-y-4 bg-indigo-50 rounded-2xl border border-indigo-100">
                  <Loader2 className="w-10 h-10 text-indigo-600 animate-spin" />
                  <p className="text-indigo-800 font-medium animate-pulse">Analyzing image and generating caption...</p>
                </div>
              )}

              {caption && (
                <div className="bg-gradient-to-br from-indigo-50 to-purple-50 p-8 rounded-2xl border border-indigo-100 shadow-inner">
                  <div className="flex items-start gap-4">
                    <div className="p-3 bg-white rounded-full shadow-sm">
                      <ImageIcon className="w-6 h-6 text-indigo-600" />
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-indigo-900 uppercase tracking-wider mb-1">Generated Caption</h3>
                      <p className="text-2xl font-medium text-slate-800 leading-relaxed capitalize">
                        "{caption}"
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {error && (
            <div className="mt-6 p-4 bg-red-50 border border-red-200 text-red-700 rounded-xl flex items-center gap-3">
              <svg className="w-5 h-5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd"></path></svg>
              <p className="text-sm font-medium">{error}</p>
            </div>
          )}

        </div>
        
        <p className="text-center text-slate-400 text-sm">
          Image Caption Generator using CNN + LSTM
        </p>
      </div>
    </div>
  );
}

export default App;
