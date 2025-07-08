import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Eye, Target, Brain, Zap } from "lucide-react";

interface DetectionTransparencyCardProps {
  classification: {
    woundType: string;
    confidence: number;
    classificationMethod: string;
    detection?: {
      confidence: number;
      measurements?: any;
      boundingBox?: any;
    };
    detectionMetadata?: {
      model: string;
      processingTime: number;
      multipleWounds: boolean;
    };
  };
}

export default function DetectionTransparencyCard({ classification }: DetectionTransparencyCardProps) {
  const hasDetectionData = classification.detection || classification.detectionMetadata;
  const detectionFound = classification.detection?.confidence > 0;
  const aiConfidence = classification.confidence || 0;
  const detectionCount = classification.detectionMetadata?.detectionCount || 0;
  
  // Calculate influence percentages
  const detectionInfluence = hasDetectionData && detectionFound ? 40 : 0;
  const aiInfluence = 100 - detectionInfluence;
  
  return (
    <Card className="mb-6 bg-blue-50 border-blue-200">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-blue-900">
          <Eye className="h-5 w-5" />
          Detection Process Transparency
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* YOLO Detection Step */}
          <div className="flex items-center justify-between p-3 bg-white rounded-lg border">
            <div className="flex items-center gap-3">
              <Target className="h-5 w-5 text-blue-600" />
              <div>
                <div className="font-medium">YOLO Detection Model</div>
                <div className="text-sm text-gray-600">
                  {classification.detectionMetadata?.model || 'smart-yolo-detection'}
                </div>
              </div>
            </div>
            <div className="text-right">
              <Badge variant={detectionFound ? 'default' : 'secondary'}>
                {detectionFound ? `${Math.round((classification.detection?.confidence || 0) * 100)}% confidence` : 
                 detectionCount === 0 ? 'No detections found' : 'No detections'}
              </Badge>
              <div className="text-xs text-gray-500 mt-1">
                {classification.detectionMetadata?.processingTime ? 
                  `${(classification.detectionMetadata.processingTime * 1000).toFixed(0)}ms` : 
                  'Processing time N/A'}
              </div>
            </div>
          </div>

          {/* AI Classification Step */}
          <div className="flex items-center justify-between p-3 bg-white rounded-lg border">
            <div className="flex items-center gap-3">
              <Brain className="h-5 w-5 text-purple-600" />
              <div>
                <div className="font-medium">AI Classification</div>
                <div className="text-sm text-gray-600">
                  {classification.classificationMethod || 'AI Vision'}
                </div>
              </div>
            </div>
            <div className="text-right">
              <Badge variant={aiConfidence > 0.8 ? 'default' : 'secondary'}>
                {Math.round(aiConfidence * 100)}% confidence
              </Badge>
              <div className="text-xs text-gray-500 mt-1">
                Classified as: {classification.woundType}
              </div>
            </div>
          </div>

          {/* Combined Results */}
          <div className="flex items-center justify-between p-3 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border border-blue-200">
            <div className="flex items-center gap-3">
              <Zap className="h-5 w-5 text-green-600" />
              <div>
                <div className="font-medium">Combined Assessment</div>
                <div className="text-sm text-gray-600">
                  {detectionInfluence > 0 ? 
                    `${detectionInfluence}% Detection + ${aiInfluence}% AI Classification` :
                    '100% AI Classification (No detection data)'}
                </div>
              </div>
            </div>
            <div className="text-right">
              <Badge variant="default">
                Final: {Math.round(aiConfidence * 100)}% confidence
              </Badge>
              <div className="text-xs text-gray-500 mt-1">
                {hasDetectionData ? 'Detection-Enhanced' : 'AI-Only'}
              </div>
            </div>
          </div>

          {/* Measurements if available */}
          {classification.detection?.measurements && (
            <div className="text-sm text-gray-600 bg-gray-50 p-2 rounded">
              <strong>Measurements:</strong> {' '}
              {classification.detection.measurements.length_mm && 
                `${classification.detection.measurements.length_mm}mm × ${classification.detection.measurements.width_mm}mm`}
              {classification.detection.measurements.area_mm2 && 
                ` (${classification.detection.measurements.area_mm2}mm²)`}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}