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
    queryKey: [`/api/my-cases/${caseId}`],
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
      case 'care_plan_generation':
        return <Zap className="h-4 w-4" />;
      default:
        return <Brain className="h-4 w-4" />;
    }
  };

  const getStepLabel = (stepType: string) => {
    switch (stepType) {
      case 'independent_classification':
        return 'Independent AI Classification';
      case 'yolo_reconsideration':
        return 'YOLO Reconsideration';
      case 'question_generation':
        return 'Question Generation';
      case 'care_plan_generation':
        return 'Care Plan Generation';
      default:
        return stepType;
    }
  };

  const formatModelName = (model: string) => {
    switch (model) {
      case 'gemini-2.5-pro':
        return 'Gemini 2.5 Pro';
      case 'gpt-4o':
        return 'GPT-4o';
      case 'gpt-3.5-turbo':
        return 'GPT-3.5 Turbo';
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
          <p className="text-sm text-gray-500 mt-2">
            Debug: Assessment {assessment ? 'found' : 'not found'}, 
            Interactions {interactions ? `found (${interactions.length})` : 'not found'}
          </p>
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

        {/* AI Interactions */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-purple-600" />
              AI Model Interactions ({interactions.length})
            </CardTitle>
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
                      <TabsTrigger value="prompt">Prompt Sent</TabsTrigger>
                      <TabsTrigger value="response">AI Response</TabsTrigger>
                      <TabsTrigger value="parsed">Parsed Result</TabsTrigger>
                    </TabsList>
                    
                    <TabsContent value="prompt" className="mt-4">
                      <ScrollArea className="h-40 w-full border rounded-md p-4">
                        <pre className="text-sm whitespace-pre-wrap text-gray-700">
                          {interaction.promptSent}
                        </pre>
                      </ScrollArea>
                    </TabsContent>
                    
                    <TabsContent value="response" className="mt-4">
                      <ScrollArea className="h-40 w-full border rounded-md p-4">
                        <pre className="text-sm whitespace-pre-wrap text-gray-700">
                          {interaction.responseReceived}
                        </pre>
                      </ScrollArea>
                    </TabsContent>
                    
                    <TabsContent value="parsed" className="mt-4">
                      <ScrollArea className="h-40 w-full border rounded-md p-4">
                        <pre className="text-sm text-gray-700">
                          {JSON.stringify(interaction.parsedResult, null, 2)}
                        </pre>
                      </ScrollArea>
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