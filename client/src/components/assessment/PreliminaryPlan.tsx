import { CheckCircle, AlertCircle, RefreshCw, ArrowRight, Edit } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import type { StepProps } from "./shared/AssessmentTypes";
import { assessmentApi } from "./shared/AssessmentUtils";

export default function PreliminaryPlan({ state, onStateChange }: StepProps) {
  const [, setLocation] = useLocation();
  const { toast } = useToast();

  // Generate final care plan mutation
  const finalPlanMutation = useMutation({
    mutationFn: async () => {
      return await assessmentApi.finalPlan(
        state.selectedImage,
        state.audience,
        state.model,
        state.aiQuestions,
        state.woundClassification,
        state.preliminaryPlan,
        state.userFeedback
      );
    },
    onSuccess: (data: any) => {
      onStateChange({
        finalCaseId: data.caseId,
        currentStep: 'final-plan'
      });
      // Navigate to the final care plan
      setLocation(`/care-plan/${data.caseId}`);
    },
    onError: (error: any) => {
      toast({
        title: "Final Plan Generation Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  // Regenerate preliminary plan mutation
  const regeneratePlanMutation = useMutation({
    mutationFn: async () => {
      return await assessmentApi.preliminaryPlan(
        state.selectedImage,
        state.audience,
        state.model,
        state.aiQuestions,
        state.woundClassification,
        state.selectedAlternative,
        state.userFeedback
      );
    },
    onSuccess: (data: any) => {
      onStateChange({ preliminaryPlan: data });
      toast({
        title: "Plan Updated",
        description: "Preliminary plan has been regenerated with your feedback."
      });
    },
    onError: (error: any) => {
      toast({
        title: "Regeneration Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const handleRegeneratePlan = () => {
    regeneratePlanMutation.mutate();
  };

  const handleGenerateFinalPlan = () => {
    finalPlanMutation.mutate();
  };

  const handleUserFeedbackChange = (feedback: string) => {
    onStateChange({ userFeedback: feedback });
  };

  if (!state.preliminaryPlan) {
    return (
      <Card className="border-2 border-medical-blue/20">
        <CardContent className="p-8 text-center">
          <div className="flex flex-col items-center space-y-4">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-medical-blue border-t-transparent"></div>
            <div className="space-y-2">
              <h3 className="text-lg font-semibold text-gray-900">Generating Preliminary Care Plan</h3>
              <p className="text-gray-600">AI is analyzing your wound assessment and creating personalized recommendations...</p>
              <div className="flex items-center justify-center space-x-2 text-sm text-gray-500">
                <div className="w-2 h-2 bg-medical-blue rounded-full animate-pulse"></div>
                <span>This may take a few moments</span>
                <div className="w-2 h-2 bg-medical-blue rounded-full animate-pulse"></div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card className="border-l-4 border-l-medical-blue shadow-md">
        <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50">
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center">
              <CheckCircle className="h-6 w-6 text-medical-blue mr-3" />
              Preliminary Assessment Complete
            </div>
            <Badge variant={state.preliminaryPlan.confidence > 0.75 ? "default" : "secondary"} className="bg-medical-blue">
              {Math.round(state.preliminaryPlan.confidence * 100)}% Confidence
            </Badge>
          </CardTitle>
          <p className="text-sm text-gray-600 mt-2">Review this preliminary assessment. Your feedback will improve the final care plan.</p>
        </CardHeader>
        <CardContent className="p-6">
          <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
            <h4 className="font-semibold text-gray-900 mb-3 flex items-center">
              <AlertCircle className="h-5 w-5 text-blue-600 mr-2" />
              Assessment Summary
            </h4>
            <div className="prose prose-sm max-w-none">
              <div className="text-gray-700 leading-relaxed whitespace-pre-wrap">{state.preliminaryPlan.assessment}</div>
            </div>
          </div>
          
          {state.preliminaryPlan.recommendations && state.preliminaryPlan.recommendations.length > 0 && (
            <div className="bg-green-50 rounded-lg p-4 border border-green-200 mt-4">
              <h4 className="font-semibold text-gray-900 mb-3 flex items-center">
                <CheckCircle className="h-5 w-5 text-green-600 mr-2" />
                Key Recommendations
              </h4>
              <ul className="space-y-3">
                {state.preliminaryPlan.recommendations.map((rec, index) => (
                  <li key={index} className="flex items-start">
                    <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">
                      <span className="text-green-600 font-bold text-xs">{index + 1}</span>
                    </div>
                    <span className="text-gray-700 leading-relaxed">{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </CardContent>
      </Card>

      {state.preliminaryPlan.needsMoreInfo && state.preliminaryPlan.additionalQuestions && (
        <Card className="border-amber-200 bg-amber-50/30">
          <CardHeader>
            <CardTitle className="flex items-center">
              <AlertCircle className="h-5 w-5 text-amber-500 mr-2" />
              Additional Questions Needed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-gray-600 mb-4">
              {state.userFeedback ? 
                "Based on your feedback, the AI needs clarification on these points:" :
                "The AI needs more information to provide a confident assessment. Please answer these additional questions:"
              }
            </p>
            <div className="space-y-4">
              {state.preliminaryPlan.additionalQuestions.map((question, index) => (
                <div key={index}>
                  <Label className="font-medium text-gray-900">{question}</Label>
                  <Textarea 
                    placeholder="Your answer..."
                    rows={2}
                    className="mt-1 border-amber-200 focus:border-amber-400"
                  />
                </div>
              ))}
            </div>
            <div className="mt-4 pt-4 border-t border-amber-200">
              <Button 
                onClick={handleRegeneratePlan}
                disabled={regeneratePlanMutation.isPending}
                className="w-full bg-amber-600 hover:bg-amber-700"
              >
                <CheckCircle className="mr-2 h-4 w-4" />
                Submit Answers & Update Plan
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      <Card className="border-amber-200 bg-amber-50/50">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center text-lg">
            <Edit className="h-5 w-5 text-amber-600 mr-2" />
            Provide Feedback (Optional)
          </CardTitle>
          <p className="text-sm text-amber-700">
            Your feedback will be integrated into the final care plan to ensure accuracy and personalization.
          </p>
        </CardHeader>
        <CardContent>
          <Textarea
            value={state.userFeedback}
            onChange={(e) => handleUserFeedbackChange(e.target.value)}
            placeholder="Add corrections, additional context, or specific concerns about the preliminary assessment..."
            rows={4}
            className="border-amber-200 focus:border-amber-400 bg-white"
          />
          
          <div className="flex gap-4 mt-4">
            <Button 
              onClick={handleRegeneratePlan}
              disabled={regeneratePlanMutation.isPending}
              variant="outline"
              className="flex-1"
            >
              {regeneratePlanMutation.isPending ? (
                <>
                  <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                  Regenerating...
                </>
              ) : (
                <>
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Regenerate Plan
                </>
              )}
            </Button>
            
            {state.preliminaryPlan.confidence > 0.75 && (
              <Button 
                onClick={handleGenerateFinalPlan}
                disabled={finalPlanMutation.isPending}
                className="flex-1 bg-medical-blue hover:bg-medical-blue/90"
              >
                {finalPlanMutation.isPending ? (
                  <>
                    <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                    Creating Final Plan...
                  </>
                ) : (
                  <>
                    Generate Complete Care Plan
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </>
                )}
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
} 