import React from 'react';

interface ProcessingPageProps {
  originalVideoSrc: string;
  roiVideos: {
    leftEye: string;
    rightEye: string;
    mouth: string;
  };
}

const ProcessingPage: React.FC<ProcessingPageProps> = ({ originalVideoSrc }) => {
  // Hard-coded ROI video paths
  const roiVideos = {
    leftEye: 'sample_data/002/002_left_eye_roi.mp4',
    mouth: 'sample_data/002/002_mouth_roi.mp4',
    nose: 'sample_data/002/002_nose_roi.mp4',
  };

  const callouts = [
    {
      name: 'Left Eye',
      description: 'ROI video of the left eye.',
      imageSrc: roiVideos.leftEye,
      imageAlt: 'Left eye video preview.',
      href: '#',
    },
    {
      name: 'Nose', 
      description: 'ROI video of the nose.',
      imageSrc: roiVideos.nose,
      imageAlt: 'Nose video preview.',
      href: '#',
    },
    {
      name: 'Mouth',
      description: 'ROI video of the mouth.',
      imageSrc: roiVideos.mouth,
      imageAlt: 'Mouth video preview.',
      href: '#',
    },
  ];

  return (
    <div className="bg-gray-100">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-2xl py-16 sm:py-24 lg:max-w-none lg:py-32">
          <h1 className="text-3xl font-bold mb-4">Processing Results</h1>

          {/* Large Preview for Original Video */}
          <div className="mb-6">
            <video
              controls
              autoPlay
              loop
              muted
              className="w-full max-w-2xl"
              src={originalVideoSrc}
            />
          </div>

          <h2 className="text-2xl font-bold text-gray-900">Regions of Interest</h2>

          <div className="mt-6 space-y-12 lg:grid lg:grid-cols-3 lg:gap-x-6 lg:space-y-0">
            {callouts.map((callout) => (
              <div key={callout.name} className="group relative">
                <video
                  controls
                  autoPlay
                  loop
                  muted
                  className="w-full rounded-lg bg-white object-cover group-hover:opacity-75 max-sm:h-80 sm:aspect-[2/1] lg:aspect-square"
                  src={callout.imageSrc}
                />
                <h3 className="mt-6 text-sm text-gray-500">
                  <a href={callout.href}>
                    <span className="absolute inset-0" />
                    {callout.name}
                  </a>
                </h3>
                <p className="text-base font-semibold text-gray-900">{callout.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProcessingPage; 