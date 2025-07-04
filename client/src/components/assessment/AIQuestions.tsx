import { CheckCircle, ArrowRight, RefreshCw } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import type { StepProps } from "./shared/AssessmentTypes";
import { assessmentApi, assessmentHelpers } from "./shared/AssessmentUtils";

export default function AIQuestions({ state, onStateChange, onNextStep }: StepProps) {
  const { toast } = useToast();

  // Update AI question answer
  const updateAnswer = (questionId: string, newAnswer: string) => {
    const updatedQuestions = assessmentHelpers.updateQuestionAnswer(
      state.aiQuestions,
      questionId,
      newAnswer
    );
    onStateChange({ aiQuestions: updatedQuestions });
  };

  // Generate preliminary care plan mutation
  const preliminaryPlanMutation = useMutation({
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
      onStateChange({
        preliminaryPlan: data,
        currentStep: 'preliminary-plan'
      });
      onNextStep();
    },
    onError: (error: any) => {
      toast({
        title: "Plan Generation Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  // Handle follow-up questions
  const handleFollowUpQuestions = async () => {
    try {
      // Store current questions as answered
      const newAnsweredQuestions = [...state.answeredQuestions, ...state.aiQuestions];
      const newRound = state.questionRound + 1;
      
      onStateChange({
        answeredQuestions: newAnsweredQuestions,
        questionRound: newRound
      });

      const data = await assessmentApi.followUpQuestions(
        state.selectedImage,
        state.audience,
        state.model,
        state.aiQuestions,
        state.woundClassification,
        newRound
      );
      
      if (data.needsMoreQuestions && data.questions && data.questions.length > 0) {
        onStateChange({ aiQuestions: data.questions });
        toast({
          title: "Follow-up Questions Generated",
          description: `Round ${data.round} questions based on your Agent Instructions.`
        });
      } else {
        // No more questions needed, proceed to care plan
        onStateChange({ 
          aiQuestions: [],
          currentStep: 'preliminary-plan'
        });
        
        // Auto-trigger preliminary plan generation
        setTimeout(() => {
          preliminaryPlanMutation.mutate();
        }, 500);
        
        toast({
          title: "Questions Complete",
          description: "Agent Instructions satisfied. Proceeding to care plan generation."
        });
      }
      
    } catch (error) {
      console.error('Follow-up questions error:', error);
      toast({
        title: "Error",
        description: "Failed to generate follow-up questions. Please try again.",
        variant: "destructive"
      });
    }
  };

  const handleProceedToPlan = () => {
    preliminaryPlanMutation.mutate();
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Step 3: Diagnostic Questions</CardTitle>
          <p className="text-gray-600">
            {state.aiQuestions.length > 0 ? (
              "The AI needs more information to improve its diagnosis. Please answer these questions:"
            ) : (
              "The AI is confident in its initial assessment. Proceeding to care plan generation."
            )}
          </p>
        </CardHeader>
      </Card>

      {state.woundClassification && (
        <Card>
          <CardHeader>
            <CardTitle>Initial Classification</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between mb-4">
              <div>
                <div className="font-medium">{state.woundClassification.woundType}</div>
                <Badge variant={state.woundClassification.confidence > 0.75 ? 'default' : 'secondary'}>
                  {Math.round(state.woundClassification.confidence * 100)}% confidence
                </Badge>
              </div>
            </div>
            
            {state.woundClassification?.alternativeTypes?.length > 0 && (
              <div className="mt-4">
                <Label>Alternative Classifications:</Label>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-2">
                  {state.woundClassification.alternativeTypes.map((alt, index) => (
                    <div
                      key={index}
                      className={`p-3 border rounded cursor-pointer transition-all ${
                        state.selectedAlternative === alt.type 
                          ? 'border-medical-blue bg-blue-50' 
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                      onClick={() => onStateChange({ selectedAlternative: alt.type })}
                    >
                      <div className="font-medium">{alt.type}</div>
                      <div className="text-sm text-gray-600">{Math.round(alt.confidence * 100)}% confidence</div>
                      <div className="text-xs text-gray-500 mt-1">{alt.reasoning}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {state.aiQuestions.map((question) => (
        <Card key={question.id}>
          <CardHeader>
            <CardTitle className="text-base">{question.question}</CardTitle>
            <Badge variant="outline">{question.category}</Badge>
          </CardHeader>
          <CardContent>
            <Textarea
              value={question.answer}
              onChange={(e) => updateAnswer(question.id, e.target.value)}
              placeholder={question.answer ? "Edit the AI's answer if needed" : "Please provide your answer..."}
              rows={3}
            />
            <div className="flex items-center mt-2 text-sm text-gray-600">
              <CheckCircle className="h-4 w-4 mr-1 text-green-500" />
              AI Confidence: {Math.round(question.confidence * 100)}%
            </div>
          </CardContent>
        </Card>
      ))}

      <Card>
        <CardContent className="pt-6">
          <Label>Additional Context (Optional)</Label>
          <Textarea
            value={state.userFeedback}
            onChange={(e) => onStateChange({ userFeedback: e.target.value })}
            placeholder="Provide any additional information about the wound or patient that might help..."
            rows={3}
            className="mt-2"
          />
          
          {state.aiQuestions.length > 0 ? (
            <div className="space-y-2 mt-4">
              {state.questionRound < 3 && (
                <Button 
                  onClick={handleFollowUpQuestions}
                  className="w-full bg-yellow-600 hover:bg-yellow-700"
                >
                  Submit Answers to Improve Assessment
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              )}
              
              <Button 
                onClick={handleProceedToPlan}
                disabled={preliminaryPlanMutation.isPending}
                variant="outline"
                className="w-full"
              >
                {preliminaryPlanMutation.isPending ? (
                  <>
                    <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                    Generating Plan...
                  </>
                ) : (
                  <>
                    Proceed with Current Assessment
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </>
                )}
              </Button>
            </div>
          ) : (
            <Button 
              onClick={handleProceedToPlan}
              disabled={preliminaryPlanMutation.isPending}
              className="w-full bg-medical-blue hover:bg-medical-blue/90 mt-4"
            >
              {preliminaryPlanMutation.isPending ? (
                <>
                  <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                  Generating Preliminary Plan...
                </>
              ) : (
                <>
                  Generate Preliminary Care Plan
                  <ArrowRight className="ml-2 h-4 w-4" />
                </>
              )}
            </Button>
          )}
        </CardContent>
      </Card>
    </div>
  );
} 