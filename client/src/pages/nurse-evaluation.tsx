import { useState, useEffect } from "react";
import { useLocation, useSearch } from "wouter";
import { ArrowLeft, Save, Star, FileText, AlertCircle, CheckCircle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { useMutation, useQuery } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

export default function NurseEvaluation() {
  const [, setLocation] = useLocation();
  const searchParams = useSearch();
  const { toast } = useToast();
  
  const [editedCarePlan, setEditedCarePlan] = useState("");
  const [rating, setRating] = useState("");
  const [nurseNotes, setNurseNotes] = useState("");
  const [agentInstructions, setAgentInstructions] = useState("");
  const [hasChanges, setHasChanges] = useState(false);

  // Extract case ID from URL params
  const caseId = new URLSearchParams(searchParams).get('caseId');

  const { data: assessmentData, isLoading } = useQuery({
    queryKey: ['/api/assessment', caseId],
    enabled: !!caseId,
    queryFn: () => fetch(`/api/assessment/${caseId}`).then(res => res.json()),
  });

  const { data: agentData } = useQuery({
    queryKey: ['/api/agents'],
    queryFn: () => fetch('/api/agents').then(res => res.json()),
  });

  useEffect(() => {
    if (assessmentData?.carePlan) {
      setEditedCarePlan(assessmentData.carePlan);
    }
  }, [assessmentData]);

  useEffect(() => {
    if (agentData?.content) {
      setAgentInstructions(agentData.content);
    }
  }, [agentData]);

  const saveEvaluationMutation = useMutation({
    mutationFn: async (data: any) => {
      return apiRequest('POST', '/api/nurse-evaluation', data);
    },
    onSuccess: () => {
      toast({
        title: "Evaluation Saved",
        description: "Nurse evaluation and instructions have been updated successfully.",
      });
      setHasChanges(false);
    },
    onError: (error: any) => {
      toast({
        title: "Save Failed",
        description: error.message || "Failed to save evaluation.",
        variant: "destructive",
      });
    },
  });

  const saveAgentInstructionsMutation = useMutation({
    mutationFn: async (content: string) => {
      return apiRequest('POST', '/api/agents', { content });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/agents'] });
    }
  });

  const handleSave = async () => {
    if (!caseId) return;

    try {
      // Save nurse evaluation
      await saveEvaluationMutation.mutateAsync({
        caseId,
        editedCarePlan,
        rating,
        nurseNotes
      });

      // Save agent instructions if changed
      if (agentInstructions !== agentData?.content) {
        await saveAgentInstructionsMutation.mutateAsync(agentInstructions);
      }
    } catch (error) {
      console.error('Save error:', error);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-bg-light flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-medical-blue mx-auto mb-3"></div>
          <p className="text-gray-600">Loading assessment...</p>
        </div>
      </div>
    );
  }

  if (!assessmentData) {
    return (
      <div className="min-h-screen bg-bg-light flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-600 mb-4">Assessment not found</p>
          <Button onClick={() => setLocation('/')}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Assessment
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-bg-light">
      {/* Header */}
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <Button 
                variant="ghost" 
                onClick={() => setLocation(`/care-plan/${caseId}`)}
                className="mr-4"
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Care Plan
              </Button>
              <div className="flex items-center">
                <FileText className="text-medical-blue text-xl mr-3" />
                <h1 className="text-xl font-semibold text-gray-900">Nurse Evaluation</h1>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <Badge variant="secondary">Case: {caseId}</Badge>
              <Button 
                onClick={handleSave}
                disabled={saveEvaluationMutation.isPending}
                className="bg-medical-blue hover:bg-blue-700"
              >
                <Save className="mr-2 h-4 w-4" />
                {saveEvaluationMutation.isPending ? 'Saving...' : 'Save Evaluation'}
              </Button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Assessment Overview */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Assessment Overview</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <span className="font-medium text-gray-700">Wound Type:</span>
                <span className="ml-2 text-gray-600">{assessmentData.classification?.woundType}</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">Stage:</span>
                <span className="ml-2 text-gray-600">{assessmentData.classification?.stage}</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">Model Used:</span>
                <span className="ml-2 text-gray-600">{assessmentData.model}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Care Plan Editing */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Care Plan Review & Edit</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <Label>Generated Care Plan</Label>
                  <Textarea
                    value={editedCarePlan}
                    onChange={(e) => {
                      setEditedCarePlan(e.target.value);
                      setHasChanges(true);
                    }}
                    rows={20}
                    className="font-mono text-sm"
                    placeholder="Review and edit the AI-generated care plan..."
                  />
                </div>
              </CardContent>
            </Card>

            {/* Rating */}
            <Card>
              <CardHeader>
                <CardTitle>Care Plan Quality Rating</CardTitle>
              </CardHeader>
              <CardContent>
                <RadioGroup value={rating} onValueChange={(value) => {
                  setRating(value);
                  setHasChanges(true);
                }}>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="excellent" id="excellent" />
                    <Label htmlFor="excellent" className="flex items-center">
                      <Star className="w-4 h-4 text-yellow-500 mr-1" />
                      Excellent - No changes needed
                    </Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="good" id="good" />
                    <Label htmlFor="good" className="flex items-center">
                      <CheckCircle className="w-4 h-4 text-green-500 mr-1" />
                      Good - Minor adjustments made
                    </Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="needs-improvement" id="needs-improvement" />
                    <Label htmlFor="needs-improvement" className="flex items-center">
                      <AlertCircle className="w-4 h-4 text-orange-500 mr-1" />
                      Needs Improvement - Significant changes required
                    </Label>
                  </div>
                </RadioGroup>
              </CardContent>
            </Card>

            {/* Nurse Notes */}
            <Card>
              <CardHeader>
                <CardTitle>Professional Notes</CardTitle>
              </CardHeader>
              <CardContent>
                <Textarea
                  value={nurseNotes}
                  onChange={(e) => {
                    setNurseNotes(e.target.value);
                    setHasChanges(true);
                  }}
                  rows={6}
                  placeholder="Add your professional observations, recommendations for AI improvement, or specific case notes..."
                />
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Agent Instructions */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>AI Agent Instructions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <p className="text-sm text-blue-800">
                      <strong>Update AI Behavior:</strong> Modify these instructions to improve future assessments based on this case.
                    </p>
                  </div>
                  
                  <Textarea
                    value={agentInstructions}
                    onChange={(e) => {
                      setAgentInstructions(e.target.value);
                      setHasChanges(true);
                    }}
                    rows={25}
                    className="font-mono text-sm"
                    placeholder="Enter AI agent instructions here..."
                  />
                  
                  <div className="text-sm text-gray-500">
                    <p>These instructions guide the AI's behavior for all future wound assessments.</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {hasChanges && (
          <Card className="mt-6">
            <CardContent className="p-4">
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <p className="text-sm text-yellow-800">
                  <strong>Unsaved Changes:</strong> You have unsaved changes. Click "Save Evaluation" to persist your updates.
                </p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}