import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useRoute } from 'wouter';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ArrowLeft, Brain, MessageSquare, Eye, Zap, AlertCircle, CheckCircle } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { queryClient } from '@/lib/queryClient';

interface AIInteraction {
  id: number;
  caseId: string;
  stepType: string;
  modelUsed: string;
  promptSent: string;
  responseReceived: string;
  parsedResult: any;
  processingTimeMs: number;
  confidenceScore: number;
  errorOccurred: boolean;
  errorMessage: string;
  createdAt: string;
}

interface WoundAssessment {
  id: number;
  caseId: string;
  caseName: string;
  woundClassification: any;
  versionNumber: number;
  createdAt: string;
  userEmail: string;
}

export default function AdminAnalysisPage() {
  const [match, params] = useRoute('/admin/analysis/:caseId');
  const caseId = params?.caseId;
  
  const { data: interactions, isLoading: interactionsLoading } = useQuery({
    queryKey: [`/api/admin/ai-interactions/${caseId}`],
    enabled: !!caseId,
  });

  const { data: assessment, isLoading: assessmentLoading } = useQuery({
    queryKey: [`/api/assessment/${caseId}`],
    enabled: !!caseId,
  });

  const getStepIcon = (stepType: string) => {
    switch (stepType) {
      case 'independent_classification':
        return <Brain className="h-4 w-4" />;
      case 'yolo_reconsideration':
        return <Eye className="h-4 w-4" />;
      case 'question_generation':
        return <MessageSquare className="h-4 w-4" />;
      case 'user_question_responses':
        return <MessageSquare className="h-4 w-4 text-blue-600" />;
      case 'care_plan_generation':
        return <Zap className="h-4 w-4" />;
      case 'care_plan_generation_fallback':
        return <Zap className="h-4 w-4 text-orange-600" />;
      default:
        return <Brain className="h-4 w-4" />;
    }
  };

  const getStepLabel = (stepType: string) => {
    switch (stepType) {
      case 'independent_classification':
        return 'Step 1: Independent AI Classification';
      case 'yolo_reconsideration':
        return 'Step 2: YOLO-Informed Reconsideration';
      case 'question_generation':
        return 'Step 3: Question Generation';
      case 'user_question_responses':
        return 'Step 4: User Question Responses';
      case 'care_plan_generation':
        return 'Step 5: Final Care Plan Generation';
      case 'care_plan_generation_fallback':
        return 'Step 5: Final Care Plan Generation (Fallback)';
      default:
        return stepType;
    }
  };

  const formatModelName = (model: string) => {
    switch (model) {
      case 'gemini-2.5-pro':
        return 'Gemini 2.5 Pro';
      case 'gemini-2.5-flash':
        return 'Gemini 2.5 Flash';
      case 'gpt-4o':
        return 'GPT-4o';
      case 'gpt-3.5-turbo':
        return 'GPT-3.5 Turbo';
      case 'user_input':
        return 'User Input';
      default:
        return model;
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const formatProcessingTime = (ms: number | null) => {
    if (!ms) return 'N/A';
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  if (interactionsLoading || assessmentLoading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading AI analysis data...</p>
        </div>
      </div>
    );
  }

  if (!assessment || !interactions || interactions.length === 0) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="text-center">
          <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <p className="text-gray-600">No analysis data found for this case.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center gap-4 mb-8">
          <Button
            variant="ghost"
            onClick={() => window.history.back()}
            className="flex items-center gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">AI Analysis Report</h1>
            <p className="text-gray-600">
              Case: {assessment.caseName || assessment.caseId} • Version {assessment.versionNumber}
            </p>
          </div>
        </div>

        {/* Case Overview */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-600" />
              Case Overview
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <p className="text-sm text-gray-500">Case ID</p>
                <p className="font-medium">{assessment.caseId}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Assessment Date</p>
                <p className="font-medium">{formatTimestamp(assessment.createdAt)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">User</p>
                <p className="font-medium">{assessment.userEmail}</p>
              </div>
            </div>
            
            {assessment.woundClassification && (
              <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                <h4 className="font-medium text-blue-900 mb-2">Final Classification</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-blue-700">Wound Type</p>
                    <p className="font-medium">{assessment.woundClassification.woundType}</p>
                  </div>
                  <div>
                    <p className="text-sm text-blue-700">Confidence</p>
                    <p className="font-medium">{Math.round(assessment.woundClassification.confidence * 100)}%</p>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Detection Engine Information */}
        {assessment.woundClassification?.detectionMetadata && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Eye className="h-5 w-5 text-green-600" />
                Detection Engine Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div className="p-4 bg-green-50 rounded-lg">
                  <h4 className="font-medium text-green-900 mb-2">Detection Method</h4>
                  <p className="text-sm text-green-700">Model: {assessment.woundClassification.detectionMetadata.model}</p>
                  <p className="text-sm text-green-700">Version: {assessment.woundClassification.detectionMetadata.version}</p>
                  <p className="text-sm text-green-700">Method: {assessment.woundClassification.detectionMetadata.methodUsed}</p>
                </div>
                
                <div className="p-4 bg-blue-50 rounded-lg">
                  <h4 className="font-medium text-blue-900 mb-2">Processing Results</h4>
                  <p className="text-sm text-blue-700">Processing Time: {assessment.woundClassification.detectionMetadata.processingTime}ms</p>
                  <p className="text-sm text-blue-700">Detections Found: {assessment.woundClassification.detectionMetadata.detectionCount}</p>
                  <p className="text-sm text-blue-700">Multiple Wounds: {assessment.woundClassification.detectionMetadata.multipleWounds ? 'Yes' : 'No'}</p>
                </div>
                
                {assessment.woundClassification.detection && (
                  <div className="p-4 bg-purple-50 rounded-lg">
                    <h4 className="font-medium text-purple-900 mb-2">Measurements</h4>
                    <p className="text-sm text-purple-700">Confidence: {Math.round((assessment.woundClassification.detection.confidence || 0) * 100)}%</p>
                    <p className="text-sm text-purple-700">Scale Calibrated: {assessment.woundClassification.detection.scaleCalibrated ? 'Yes' : 'No'}</p>
                    {assessment.woundClassification.detection.measurements && (
                      <>
                        <p className="text-sm text-purple-700">Area: {Math.round(assessment.woundClassification.detection.measurements.areaMm2)}mm²</p>
                        <p className="text-sm text-purple-700">Length: {Math.round(assessment.woundClassification.detection.measurements.lengthMm)}mm</p>
                        <p className="text-sm text-purple-700">Width: {Math.round(assessment.woundClassification.detection.measurements.widthMm)}mm</p>
                      </>
                    )}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* AI Interactions */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-purple-600" />
              AI Model Interactions ({interactions.length})
            </CardTitle>
            <div className="text-sm text-gray-600 mt-2">
              Complete workflow showing: Initial AI classification → YOLO detection analysis → Question generation → User responses → Final care plan generation
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {interactions.map((interaction: AIInteraction, index: number) => (
                <div key={interaction.id} className="border rounded-lg p-6 bg-white">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      {getStepIcon(interaction.stepType)}
                      <h3 className="font-semibold text-lg">{getStepLabel(interaction.stepType)}</h3>
                      <Badge variant="outline">Step {index + 1}</Badge>
                    </div>
                    <div className="flex items-center gap-4 text-sm text-gray-500">
                      <span>{formatModelName(interaction.modelUsed)}</span>
                      <span>{formatTimestamp(interaction.createdAt)}</span>
                      <span>{formatProcessingTime(interaction.processingTimeMs)}</span>
                      {interaction.confidenceScore && (
                        <Badge variant={interaction.confidenceScore > 80 ? 'default' : 'secondary'}>
                          {interaction.confidenceScore}% confidence
                        </Badge>
                      )}
                    </div>
                  </div>

                  <Tabs defaultValue="prompt" className="w-full">
                    <TabsList className="grid w-full grid-cols-3">
                      <TabsTrigger value="prompt">
                        Full Prompt ({interaction.promptSent.length.toLocaleString()} chars)
                      </TabsTrigger>
                      <TabsTrigger value="response">
                        AI Response ({interaction.responseReceived.length.toLocaleString()} chars)
                      </TabsTrigger>
                      <TabsTrigger value="parsed">
                        Parsed Result
                      </TabsTrigger>
                    </TabsList>
                    
                    <TabsContent value="prompt" className="mt-4">
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 text-sm text-gray-500">
                          <span>Character Count: {interaction.promptSent.length.toLocaleString()}</span>
                          <span>•</span>
                          <span>Image Included: {interaction.promptSent.includes('[IMAGE PROVIDED:') ? 'Yes' : 'No'}</span>
                        </div>
                        <ScrollArea className="h-64 w-full border rounded-md p-4">
                          <pre className="text-sm whitespace-pre-wrap text-gray-700 leading-relaxed">
                            {interaction.promptSent}
                          </pre>
                        </ScrollArea>
                      </div>
                    </TabsContent>
                    
                    <TabsContent value="response" className="mt-4">
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 text-sm text-gray-500">
                          <span>Character Count: {interaction.responseReceived.length.toLocaleString()}</span>
                          <span>•</span>
                          <span>Response Type: {interaction.stepType === 'user_question_responses' ? 'User Input' : 'AI Generated'}</span>
                        </div>
                        <ScrollArea className="h-64 w-full border rounded-md p-4">
                          <pre className="text-sm whitespace-pre-wrap text-gray-700 leading-relaxed">
                            {interaction.responseReceived}
                          </pre>
                        </ScrollArea>
                      </div>
                    </TabsContent>
                    
                    <TabsContent value="parsed" className="mt-4">
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 text-sm text-gray-500">
                          <span>Data Type: {interaction.parsedResult ? typeof interaction.parsedResult : 'None'}</span>
                          {interaction.parsedResult && (
                            <>
                              <span>•</span>
                              <span>Keys: {Object.keys(interaction.parsedResult).join(', ')}</span>
                            </>
                          )}
                        </div>
                        <ScrollArea className="h-64 w-full border rounded-md p-4">
                          <pre className="text-sm text-gray-700 leading-relaxed">
                            {interaction.parsedResult ? JSON.stringify(interaction.parsedResult, null, 2) : 'No parsed result available'}
                          </pre>
                        </ScrollArea>
                      </div>
                    </TabsContent>
                  </Tabs>

                  {interaction.errorOccurred && (
                    <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
                      <div className="flex items-center gap-2 text-red-700">
                        <AlertCircle className="h-4 w-4" />
                        <span className="font-medium">Error Occurred</span>
                      </div>
                      <p className="text-sm text-red-600 mt-1">{interaction.errorMessage}</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}