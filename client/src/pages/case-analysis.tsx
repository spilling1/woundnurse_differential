import { useState, useEffect } from 'react';
import { useRoute } from 'wouter';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ArrowLeft, Eye, Brain, MessageSquare, AlertCircle } from 'lucide-react';
import { Link } from 'wouter';
import { useAuth } from '@/hooks/useAuth';

interface DetectionData {
  detections?: Array<{
    boundingBox: { x: number; y: number; width: number; height: number };
    confidence: number;
    measurements: { lengthMm: number; widthMm: number; areaMm2: number };
  }>;
  model?: string;
  version?: string;
  processingTime?: number;
}

interface Classification {
  woundType: string;
  confidence: number;
  classificationMethod: string;
  modelInfo: {
    type: string;
    accuracy: string;
    apiCall?: boolean;
  };
  additionalObservations?: string;
  location?: string;
  size?: string;
  detectionMetadata?: {
    model: string;
    version: string;
    processingTime: number;
  };
}

interface QuestionData {
  id: number;
  question: string;
  answer: string;
  questionType: string;
  isAnswered: boolean;
  createdAt: string;
  answeredAt?: string;
}

interface CaseAnalysisData {
  caseId: string;
  caseName: string;
  model: string;
  classification: Classification;
  detectionData: DetectionData | null;
  questions: QuestionData[];
  contextData: any;
  createdAt: string;
}

export default function CaseAnalysis() {
  const [match, params] = useRoute<{ caseId: string }>('/case-analysis/:caseId');
  const [analysisData, setAnalysisData] = useState<CaseAnalysisData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { user } = useAuth();

  useEffect(() => {
    if (match && params?.caseId) {
      fetchAnalysisData(params.caseId);
    }
  }, [match, params?.caseId]);

  const fetchAnalysisData = async (caseId: string) => {
    try {
      setLoading(true);
      
      // Fetch case data
      const caseResponse = await fetch(`/api/assessment/${caseId}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!caseResponse.ok) {
        throw new Error('Failed to fetch case data');
      }
      
      const caseData = await caseResponse.json();
      
      // Fetch questions data
      const questionsResponse = await fetch(`/api/questions/${caseId}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      const questionsData = questionsResponse.ok ? await questionsResponse.json() : [];
      
      setAnalysisData({
        caseId: caseData.caseId,
        caseName: caseData.caseName || 'Unnamed Case',
        model: caseData.model,
        classification: typeof caseData.classification === 'string' 
          ? JSON.parse(caseData.classification) 
          : caseData.classification,
        detectionData: caseData.detectionData || null,
        questions: questionsData,
        contextData: typeof caseData.contextData === 'string' 
          ? JSON.parse(caseData.contextData) 
          : caseData.contextData,
        createdAt: caseData.createdAt
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch analysis data');
    } finally {
      setLoading(false);
    }
  };

  if (!match || !params?.caseId) {
    return <div>Case not found</div>;
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center justify-center h-64">
            <div className="text-lg">Loading case analysis...</div>
          </div>
        </div>
      </div>
    );
  }

  if (error || !analysisData) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-4xl mx-auto">
          <div className="text-red-600 text-center">
            Error: {error || 'Failed to load case analysis'}
          </div>
        </div>
      </div>
    );
  }

  const { classification, detectionData, questions, model, caseId, caseName } = analysisData;

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <Link href="/my-cases">
            <Button variant="outline" className="mb-4">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to My Cases
            </Button>
          </Link>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Detailed Case Analysis
          </h1>
          <p className="text-gray-600">
            {caseName} (Case ID: {caseId})
          </p>
        </div>

        {/* AI Model Used */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Brain className="w-5 h-5 mr-2" />
              AI Model Analysis
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h3 className="font-semibold mb-2">Primary AI Model</h3>
                <Badge variant="secondary" className="text-sm">
                  {model}
                </Badge>
                <p className="text-sm text-gray-600 mt-2">
                  Classification Method: {classification.classificationMethod}
                </p>
              </div>
              <div>
                <h3 className="font-semibold mb-2">Analysis Confidence</h3>
                <div className="flex items-center">
                  <span className="text-2xl font-bold text-green-600">
                    {Math.round((classification.confidence || 0) * 100)}%
                  </span>
                  <span className="text-gray-500 ml-2">confidence</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* YOLO Detection Analysis */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Eye className="w-5 h-5 mr-2" />
              YOLO Detection Analysis
            </CardTitle>
          </CardHeader>
          <CardContent>
            {detectionData && detectionData.detections && detectionData.detections.length > 0 ? (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h3 className="font-semibold mb-2">Detection Model</h3>
                    <Badge variant="outline">
                      {detectionData.model || 'Unknown'}
                    </Badge>
                    <p className="text-sm text-gray-600 mt-1">
                      Version: {detectionData.version || 'Unknown'}
                    </p>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-2">Processing Time</h3>
                    <span className="text-lg font-mono">
                      {detectionData.processingTime || 'Unknown'}ms
                    </span>
                  </div>
                </div>
                
                <div>
                  <h3 className="font-semibold mb-2">Wound Detections</h3>
                  {detectionData.detections.map((detection, index) => (
                    <div key={index} className="bg-gray-50 p-4 rounded-lg mb-3">
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                          <p className="text-sm font-medium">Detection Confidence</p>
                          <p className="text-lg font-bold text-blue-600">
                            {Math.round(detection.confidence * 100)}%
                          </p>
                        </div>
                        <div>
                          <p className="text-sm font-medium">Bounding Box</p>
                          <p className="text-sm text-gray-600">
                            {detection.boundingBox.x}, {detection.boundingBox.y}, 
                            {detection.boundingBox.width} × {detection.boundingBox.height}
                          </p>
                        </div>
                        <div>
                          <p className="text-sm font-medium">Measurements</p>
                          <p className="text-sm text-gray-600">
                            {detection.measurements.lengthMm.toFixed(1)} × {detection.measurements.widthMm.toFixed(1)} mm
                          </p>
                          <p className="text-sm text-gray-600">
                            Area: {detection.measurements.areaMm2.toFixed(1)} mm²
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <AlertCircle className="w-12 h-12 text-amber-500 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  No YOLO Detection Data Available
                </h3>
                <p className="text-gray-600 mb-4">
                  This case was processed without YOLO wound detection. The system may have:
                </p>
                <ul className="text-sm text-gray-600 text-left max-w-md mx-auto space-y-1">
                  <li>• Used AI vision models only (Gemini/GPT)</li>
                  <li>• Encountered YOLO service unavailability</li>
                  <li>• Processed before YOLO integration was active</li>
                </ul>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Questions & Impact Analysis */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center">
              <MessageSquare className="w-5 h-5 mr-2" />
              Questions & Impact Analysis
            </CardTitle>
          </CardHeader>
          <CardContent>
            {questions && questions.length > 0 ? (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h3 className="font-semibold text-blue-900 mb-2">Questions Asked</h3>
                    <p className="text-2xl font-bold text-blue-600">
                      {questions.length}
                    </p>
                  </div>
                  <div className="bg-green-50 p-4 rounded-lg">
                    <h3 className="font-semibold text-green-900 mb-2">Answered</h3>
                    <p className="text-2xl font-bold text-green-600">
                      {questions.filter(q => q.isAnswered).length}
                    </p>
                  </div>
                  <div className="bg-amber-50 p-4 rounded-lg">
                    <h3 className="font-semibold text-amber-900 mb-2">Skipped</h3>
                    <p className="text-2xl font-bold text-amber-600">
                      {questions.filter(q => !q.isAnswered).length}
                    </p>
                  </div>
                </div>
                
                <div>
                  <h3 className="font-semibold mb-4">Question Details</h3>
                  {questions.map((question) => (
                    <div key={question.id} className="bg-gray-50 p-4 rounded-lg mb-3">
                      <div className="mb-2">
                        <Badge variant={question.isAnswered ? "default" : "secondary"}>
                          {question.questionType}
                        </Badge>
                      </div>
                      <p className="font-medium mb-2">{question.question}</p>
                      {question.isAnswered && question.answer ? (
                        <p className="text-gray-600 bg-white p-3 rounded border">
                          <strong>Answer:</strong> {question.answer}
                        </p>
                      ) : (
                        <p className="text-gray-500 italic">No answer provided</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <AlertCircle className="w-12 h-12 text-amber-500 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  No Questions Were Asked
                </h3>
                <p className="text-gray-600 mb-4">
                  This case was processed without any diagnostic questions. This typically happens when:
                </p>
                <ul className="text-sm text-gray-600 text-left max-w-md mx-auto space-y-1">
                  <li>• AI confidence was above 80% from visual analysis alone</li>
                  <li>• The system determined no additional information was needed</li>
                  <li>• The case was processed in direct care plan generation mode</li>
                </ul>
              </div>
            )}
          </CardContent>
        </Card>

        {/* AI Classification Results */}
        <Card>
          <CardHeader>
            <CardTitle>AI Classification Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h3 className="font-semibold mb-2">Wound Type</h3>
                  <Badge variant="destructive" className="text-sm">
                    {classification.woundType}
                  </Badge>
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Location</h3>
                  <p className="text-sm text-gray-600">
                    {classification.location || 'Not specified'}
                  </p>
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Size Assessment</h3>
                  <p className="text-sm text-gray-600">
                    {classification.size || 'Not specified'}
                  </p>
                </div>
              </div>
              
              {classification.additionalObservations && (
                <div>
                  <h3 className="font-semibold mb-2">Additional Observations</h3>
                  <p className="text-sm text-gray-600 bg-gray-50 p-3 rounded">
                    {classification.additionalObservations}
                  </p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}