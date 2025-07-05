import React, { useRef, useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Ruler, Eye, Target, AlertTriangle } from 'lucide-react';

interface WoundDetectionData {
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  confidence: number;
  measurements: {
    lengthMm: number;
    widthMm: number;
    areaMm2: number;
  };
  referenceObjectDetected: boolean;
  scaleCalibrated: boolean;
}

interface WoundVisualizationProps {
  imageData: string;
  detectionData?: {
    detections: WoundDetectionData[];
    model: string;
    version: string;
    processingTime: number;
  };
  classification?: any;
}

export default function WoundVisualization({
  imageData,
  detectionData,
  classification
}: WoundVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);

  useEffect(() => {
    if (!canvasRef.current || !imageData) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      // Set canvas size to match image
      canvas.width = img.width;
      canvas.height = img.height;
      
      // Draw the image
      ctx.drawImage(img, 0, 0);
      
      // Draw detection overlays if available
      if (detectionData && detectionData.detections.length > 0 && showOverlay) {
        drawDetectionOverlays(ctx, detectionData.detections, img.width, img.height);
      }
      
      setImageLoaded(true);
    };
    
    img.src = `data:image/jpeg;base64,${imageData}`;
  }, [imageData, detectionData, showOverlay]);

  const drawDetectionOverlays = (
    ctx: CanvasRenderingContext2D,
    detections: WoundDetectionData[],
    imgWidth: number,
    imgHeight: number
  ) => {
    detections.forEach((detection, index) => {
      const { boundingBox, confidence, scaleCalibrated } = detection;
      
      // Calculate actual coordinates
      const x = boundingBox.x;
      const y = boundingBox.y;
      const width = boundingBox.width;
      const height = boundingBox.height;
      
      // Set overlay style based on confidence
      const alpha = Math.max(0.3, confidence);
      const color = confidence > 0.7 ? 'rgba(34, 197, 94, ' : 'rgba(251, 191, 36, ';
      
      // Draw bounding box
      ctx.strokeStyle = color + alpha + ')';
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);
      
      // Draw filled background for better visibility
      ctx.fillStyle = color + '0.1)';
      ctx.fillRect(x, y, width, height);
      
      // Draw confidence badge
      ctx.fillStyle = color + '0.9)';
      ctx.fillRect(x, y - 25, 80, 20);
      
      ctx.fillStyle = 'white';
      ctx.font = '12px Arial';
      ctx.fillText(`${Math.round(confidence * 100)}%`, x + 5, y - 10);
      
      // Draw calibration indicator
      if (scaleCalibrated) {
        ctx.fillStyle = 'rgba(34, 197, 94, 0.9)';
        ctx.fillRect(x + width - 60, y - 25, 55, 20);
        ctx.fillStyle = 'white';
        ctx.fillText('Scaled', x + width - 55, y - 10);
      }
    });
  };

  const primaryDetection = detectionData?.detections[0];
  const hasDetectionData = detectionData && detectionData.detections.length > 0;

  return (
    <div className="space-y-6">
      {/* Image with Detection Overlay */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Eye className="h-5 w-5" />
              Wound Analysis
            </CardTitle>
            {hasDetectionData && (
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setShowOverlay(!showOverlay)}
                  className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200"
                >
                  {showOverlay ? 'Hide' : 'Show'} Detection
                </button>
                <Badge variant="secondary">
                  {detectionData.detections.length} wound{detectionData.detections.length > 1 ? 's' : ''} detected
                </Badge>
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="relative">
            <canvas
              ref={canvasRef}
              className="max-w-full h-auto border border-gray-200 rounded-lg"
              style={{ maxHeight: '500px' }}
            />
            {!imageLoaded && (
              <div className="absolute inset-0 flex items-center justify-center bg-gray-100 rounded-lg">
                <div className="text-gray-500">Loading image...</div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Detection Data */}
      {hasDetectionData && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Measurements */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Ruler className="h-5 w-5" />
                Wound Measurements
              </CardTitle>
            </CardHeader>
            <CardContent>
              {primaryDetection ? (
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Length:</span>
                    <span className="font-mono">
                      {primaryDetection.measurements.lengthMm.toFixed(1)} mm
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Width:</span>
                    <span className="font-mono">
                      {primaryDetection.measurements.widthMm.toFixed(1)} mm
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Area:</span>
                    <span className="font-mono">
                      {primaryDetection.measurements.areaMm2.toFixed(1)} mm²
                    </span>
                  </div>
                  <div className="pt-2 border-t">
                    <div className="flex items-center gap-2 text-sm">
                      {primaryDetection.scaleCalibrated ? (
                        <Badge variant="default" className="bg-green-100 text-green-800">
                          Scale Calibrated
                        </Badge>
                      ) : (
                        <Badge variant="secondary" className="bg-yellow-100 text-yellow-800">
                          Estimated Scale
                        </Badge>
                      )}
                      {primaryDetection.referenceObjectDetected && (
                        <Badge variant="outline">Reference Object Found</Badge>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-sm text-gray-500">No measurements available</div>
              )}
            </CardContent>
          </Card>

          {/* Detection Quality */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                Detection Quality
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Confidence:</span>
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full ${
                          (primaryDetection?.confidence || 0) > 0.7 
                            ? 'bg-green-500' 
                            : 'bg-yellow-500'
                        }`}
                        style={{ width: `${((primaryDetection?.confidence || 0) * 100)}%` }}
                      />
                    </div>
                    <span className="font-mono text-sm">
                      {Math.round((primaryDetection?.confidence || 0) * 100)}%
                    </span>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Model:</span>
                  <span className="font-mono text-sm">{detectionData.model}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Processing Time:</span>
                  <span className="font-mono text-sm">{detectionData.processingTime}ms</span>
                </div>
                
                {(primaryDetection?.confidence || 0) < 0.7 && (
                  <div className="flex items-start gap-2 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <AlertTriangle className="h-4 w-4 text-yellow-600 mt-0.5" />
                    <div className="text-sm text-yellow-800">
                      Lower confidence detection. Consider retaking the image with better lighting or different angle.
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Enhanced Classification Display */}
      {classification && (
        <Card>
          <CardHeader>
            <CardTitle>Enhanced Classification Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="text-sm font-medium">Wound Type:</div>
                <div className="text-lg">{classification.woundType}</div>
              </div>
              <div className="space-y-2">
                <div className="text-sm font-medium">Stage:</div>
                <div className="text-lg">{classification.stage}</div>
              </div>
              <div className="space-y-2">
                <div className="text-sm font-medium">Location:</div>
                <div className="text-lg">{classification.location}</div>
              </div>
              <div className="space-y-2">
                <div className="text-sm font-medium">Size Category:</div>
                <div className="text-lg capitalize">{classification.size}</div>
              </div>
              {classification.preciseMeasurements && (
                <div className="md:col-span-2 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="text-sm font-medium mb-2">Precise Measurements:</div>
                  <div className="text-sm text-blue-800">
                    {classification.preciseMeasurements.lengthMm.toFixed(1)} mm × {classification.preciseMeasurements.widthMm.toFixed(1)} mm 
                    (Area: {classification.preciseMeasurements.areaMm2.toFixed(1)} mm²)
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}