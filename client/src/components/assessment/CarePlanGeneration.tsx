import { useState, useEffect } from "react";
import { RefreshCw, CheckCircle, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useMutation } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import type { StepProps } from "./shared/AssessmentTypes";
import { assessmentApi } from "./shared/AssessmentUtils";

export default function CarePlanGeneration({ state, onStateChange, onNextStep }: StepProps) {
  const { toast } = useToast();
  const [progress, setProgress] = useState(0);
  const [generatedPlan, setGeneratedPlan] = useState<any>(null);

  // Generate final care plan mutation
  const finalPlanMutation = useMutation({
    mutationFn: async () => {
      return await assessmentApi.finalPlan(
        state.selectedImage,
        state.audience,
        state.model,
        state.aiQuestions,
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

  // Auto-start plan generation when component mounts
  useEffect(() => {
    if (!finalPlanMutation.isPending && !finalPlanMutation.isSuccess && !finalPlanMutation.isError) {
      finalPlanMutation.mutate();
    }
  }, []);

  const handleViewCarePlan = () => {
    onNextStep();
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

  return null;
}