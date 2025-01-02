'use client'

import React, { useState, useEffect } from 'react';
import FileUpload from '../components/FileUpload';
import AugmentedVideo from './AugmentedVideo';
import LipVideo from './LipVideo';


const MainPage: React.FC = () => {
  const [result, setResult] = useState<boolean | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [isProcessing, setIsProcessing] = useState(false);
  const [originalVideoSrc, setOriginalVideoSrc] = useState<string | null>(null);
  const [lipVideoUrl, lipSetVideoUrl] = useState<string | null>(null);
  const [lipScore, lipSetScore] = useState<string | null>(null);
  const [mmnetVideoUrl, mmnetSetVideoUrl] = useState<string | null>(null);
  const [mmnetScore, mmnetSetScore] = useState<string | null>(null);
  const [ppgVideos, setPpgVideos] = useState<{ ppgGraphUrl: string; ppgMaskUrl: string; ppgTransformedUrl: string } | null>(null);



  const handleFileUpload = async (file: File) => {
    setIsProcessing(true);
    setResult(null);
    setOriginalVideoSrc(URL.createObjectURL(file));
  };



  useEffect(() => {
    if (originalVideoSrc && result !== null) {
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
    }
  }, [originalVideoSrc, result]);


  return (
    <main className="flex flex-col items-center justify-center p-6">
      <h1 className="text-3xl font-bold mt-24 mb-4">Deepfake Detector</h1>
      <FileUpload
        onFileUpload={handleFileUpload}
        lipSetVideoUrl={lipSetVideoUrl}
        lipSetScore={lipSetScore}
        mmnetSetScore={mmnetSetScore}
        mmnetSetVideoUrl={mmnetSetVideoUrl}
        setPpgVideos={setPpgVideos}
      />
      <p className="mt-2 flex-col mb-64 text-sm text-gray-600">Upload an image or video to check for deepfakes.</p>
      {mmnetVideoUrl && <AugmentedVideo videoUrl={mmnetVideoUrl} score={mmnetScore} />}
      {lipVideoUrl && <LipVideo videoUrl={lipVideoUrl} score={lipScore} />}
      {ppgVideos && (
                            <div className='flex flex-col items-center justify-center'>
                                <video autoPlay loop muted src={ppgVideos.ppgGraphUrl} />
                                <video autoPlay loop muted src={ppgVideos.ppgMaskUrl} />
                                <video autoPlay loop muted src={ppgVideos.ppgTransformedUrl} />
                            </div>
                        )}
    </main>
  );
};

export default MainPage;