import { useState, useEffect } from "react";
import { RefreshCw, CheckCircle, ArrowRight, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

import { useMutation } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { useLocation } from "wouter";
import type { StepProps } from "./shared/AssessmentTypes";
import { assessmentApi } from "./shared/AssessmentUtils";

export default function CarePlanGeneration({ state, onStateChange, onNextStep }: StepProps) {
  const { toast } = useToast();
  const [, setLocation] = useLocation();
  const [progress, setProgress] = useState(0);
  const [generatedPlan, setGeneratedPlan] = useState<any>(null);
  const [showDuplicateDialog, setShowDuplicateDialog] = useState(!!state.duplicateInfo);
  const [resolvingDuplicate, setResolvingDuplicate] = useState(false);

  // Generate final care plan mutation
  const finalPlanMutation = useMutation({
    mutationFn: async () => {
      console.log('CarePlanGeneration: state.model =', state.model);
      console.log('CarePlanGeneration: state.audience =', state.audience);
      
      // Ensure we have a valid model, fallback to gemini-2.5-pro if needed
      const modelToUse = state.model || 'gemini-2.5-pro';
      console.log('CarePlanGeneration: Using model =', modelToUse);
      
      return await assessmentApi.finalPlan(
        state.selectedImage,
        state.audience,
        modelToUse,
        state.answeredQuestions || [],
        state.woundClassification,
        state.userFeedback
      );
    },
    onSuccess: (data: any) => {
      setGeneratedPlan(data);
      onStateChange({
        finalCaseId: data.caseId
      });
      
      toast({
        title: "Care Plan Generated",
        description: "Your personalized care plan is ready for review.",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Plan Generation Failed",
        description: error.message,
        variant: "destructive",
      });
    }
  });

  // Progress bar animation
  useEffect(() => {
    if (finalPlanMutation.isPending) {
      const interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) return prev;
          return prev + Math.random() * 15;
        });
      }, 500);

      return () => clearInterval(interval);
    } else if (finalPlanMutation.isSuccess) {
      setProgress(100);
    }
  }, [finalPlanMutation.isPending, finalPlanMutation.isSuccess]);

  // Auto-start plan generation when component mounts (only if no duplicate detected)
  useEffect(() => {
    if (!state.duplicateInfo && !finalPlanMutation.isPending && !finalPlanMutation.isSuccess && !finalPlanMutation.isError) {
      finalPlanMutation.mutate();
    }
  }, [state.duplicateInfo]);

  const handleViewCarePlan = () => {
    if (generatedPlan?.caseId) {
      // Navigate to the care plan page for this case
      setLocation(`/care-plan/${generatedPlan.caseId}`);
    }
  };

  // Handlers for duplicate detection
  const handleCreateFollowUp = async () => {
    setShowDuplicateDialog(false);
    setResolvingDuplicate(true);
    
    try {
      // Create follow-up assessment with existing case ID
      // Use classification from existing case since we detected duplicate early
      const modelToUse: ModelType = state.model || 'gemini-2.5-pro';
      const classificationToUse = state.woundClassification || state.duplicateInfo?.existingCase?.classification;
      
      const followUpResponse = await assessmentApi.finalPlan(
        state.selectedImage,
        state.audience,
        modelToUse,
        state.aiQuestions || [],
        classificationToUse,
        state.userFeedback || '',
        state.duplicateInfo?.existingCase?.caseId // Pass existing case ID
      );
      
      setGeneratedPlan(followUpResponse);
      onStateChange({
        finalCaseId: followUpResponse.caseId
      });
      
      toast({
        title: "Follow-Up Assessment Created",
        description: `Added as version ${followUpResponse.versionNumber} to case ${followUpResponse.caseId}`,
      });
    } catch (error) {
      toast({
        title: "Error Creating Follow-Up",
        description: "Failed to create follow-up assessment. Please try again.",
        variant: "destructive",
      });
    } finally {
      setResolvingDuplicate(false);
    }
  };

  const handleCreateNewCase = async () => {
    setShowDuplicateDialog(false);
    setResolvingDuplicate(true);
    
    try {
      // Force create new case (backend will ignore duplicate detection)
      // Use classification from existing case since we detected duplicate early
      const modelToUse: ModelType = state.model || 'gemini-2.5-pro';
      const classificationToUse = state.woundClassification || state.duplicateInfo?.existingCase?.classification;
      
      const newCaseResponse = await assessmentApi.finalPlan(
        state.selectedImage,
        state.audience,
        modelToUse,
        state.aiQuestions || [],
        classificationToUse,
        state.userFeedback || '',
        null, // No existing case ID - force new case
        true  // forceNew flag
      );
      
      setGeneratedPlan(newCaseResponse);
      onStateChange({
        finalCaseId: newCaseResponse.caseId
      });
      
      toast({
        title: "New Case Created",
        description: "Created new assessment case with the same image.",
      });
    } catch (error) {
      toast({
        title: "Error Creating New Case",
        description: "Failed to create new assessment case. Please try again.",
        variant: "destructive",
      });
    } finally {
      setResolvingDuplicate(false);
    }
  };

  if (finalPlanMutation.isPending) {
    return (
      <div className="space-y-6">
        <Card>
          <CardHeader className="text-center">
            <CardTitle className="flex items-center justify-center gap-2">
              <RefreshCw className="h-5 w-5 animate-spin text-medical-blue" />
              Generating Your Care Plan
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="text-center text-gray-600">
              <p className="mb-4">
                Our AI is analyzing your wound assessment and creating a personalized care plan
                tailored for {state.audience === 'family' ? 'family caregivers' : state.audience === 'patient' ? 'patient self-care' : 'medical professionals'}.
              </p>
              
              {state.model?.includes('gemini') && (
                <p className="mb-4 text-sm text-blue-600 bg-blue-50 p-3 rounded-lg">
                  <strong>Note:</strong> Gemini analysis can take up to 60 seconds for thorough medical image processing. Please be patient while we generate your detailed care plan.
                </p>
              )}
              
              <div className="space-y-2">
                <Progress value={progress} className="w-full" />
                <p className="text-sm text-gray-500">
                  {progress < 30 && "Analyzing wound characteristics..."}
                  {progress >= 30 && progress < 60 && "Generating treatment recommendations..."}
                  {progress >= 60 && progress < 90 && "Customizing for your audience..."}
                  {progress >= 90 && "Finalizing your care plan..."}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (finalPlanMutation.isError) {
    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-red-600">Plan Generation Failed</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-gray-600 mb-4">
              There was an error generating your care plan. Please try again.
            </p>
            <Button 
              onClick={() => finalPlanMutation.mutate()}
              className="w-full"
            >
              Retry Plan Generation
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Show duplicate resolution status
  if (showDuplicateDialog && !resolvingDuplicate) {
    return (
      <div className="space-y-6">
        <Card>
          <CardHeader className="text-center">
            <CardTitle className="flex items-center justify-center gap-2 text-amber-600">
              <AlertTriangle className="h-5 w-5" />
              Duplicate Image Detected
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="text-center">
              <p className="text-gray-600 mb-4">
                We found an identical image in your previous assessments. Please choose how to proceed:
              </p>
              
              <div className="bg-amber-50 rounded-lg p-4 mb-4">
                <div className="text-sm text-gray-700">
                  <strong>Existing Case:</strong> {state.duplicateInfo?.existingCase?.caseId}
                  <br />
                  <strong>Created:</strong> {state.duplicateInfo?.existingCase?.createdAt 
                    ? new Date(state.duplicateInfo.existingCase.createdAt).toLocaleDateString() 
                    : 'Unknown'}
                </div>
              </div>
              
              <div className="flex flex-col gap-2">
                <Button 
                  onClick={handleCreateFollowUp}
                  className="w-full bg-medical-blue hover:bg-medical-blue/90"
                >
                  Create Follow-Up Assessment
                </Button>
                <Button 
                  onClick={handleCreateNewCase}
                  variant="outline"
                  className="w-full"
                >
                  Create New Case Anyway
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Show resolving duplicate status
  if (resolvingDuplicate) {
    return (
      <div className="space-y-6">
        <Card>
          <CardHeader className="text-center">
            <CardTitle className="flex items-center justify-center gap-2 text-medical-blue">
              <RefreshCw className="h-5 w-5 animate-spin" />
              Processing Your Choice
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="text-center">
              <p className="text-gray-600 mb-4">
                Processing your duplicate image resolution choice...
              </p>
              
              <div className="space-y-2">
                <Progress value={80} className="w-full" />
                <p className="text-sm text-gray-500">
                  Creating your assessment case
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (finalPlanMutation.isSuccess && generatedPlan) {
    return (
      <div className="space-y-6">
        <Card>
          <CardHeader className="text-center">
            <CardTitle className="flex items-center justify-center gap-2 text-green-600">
              <CheckCircle className="h-5 w-5" />
              Care Plan Generated Successfully
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="text-center">
              <p className="text-gray-600 mb-6">
                Your personalized care plan has been created and saved as case <strong>{generatedPlan.caseId}</strong>.
              </p>
              
              {/* Care Plan Preview */}
              <div className="bg-gray-50 rounded-lg p-4 text-left">
                <h4 className="font-semibold text-gray-800 mb-2">Care Plan Preview:</h4>
                <div className="text-sm text-gray-600 space-y-2">
                  {generatedPlan.plan?.split('\n\n')[0] && (
                    <p className="line-clamp-3">
                      {generatedPlan.plan.split('\n\n')[0]}...
                    </p>
                  )}
                  <p className="text-xs text-gray-500 italic">
                    Click below to view the complete care plan with detailed instructions and recommendations.
                  </p>
                </div>
              </div>
              
              <Button 
                onClick={handleViewCarePlan}
                className="w-full bg-medical-blue hover:bg-medical-blue/90 mt-4"
              >
                View Complete Care Plan
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Default loading state - this shouldn't normally be reached
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="text-center">
          <CardTitle className="flex items-center justify-center gap-2">
            <RefreshCw className="h-5 w-5 animate-spin text-medical-blue" />
            Processing Assessment
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-gray-600">
            <p>Processing your wound assessment...</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}