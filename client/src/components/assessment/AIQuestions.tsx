import { CheckCircle, ArrowRight, RefreshCw, Camera, Upload } from "lucide-react";
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

  // Helper function to get user-friendly detection method names
  const getDetectionMethodName = (model: string): string => {
    switch (model) {
      case 'yolo9':
      case 'yolo8':
      case 'yolov8':
        return 'Smart Wound Detection';
      case 'color-detection':
        return 'Color-based Detection';
      case 'google-cloud-vision':
        return 'Google Cloud Vision';
      case 'azure-computer-vision':
        return 'Azure Computer Vision';
      case 'enhanced-fallback':
        return 'Enhanced Image Analysis';
      default:
        return 'Image Analysis';
    }
  };

  // Update AI question answer
  const updateAnswer = (questionId: string, newAnswer: string) => {
    const updatedQuestions = assessmentHelpers.updateQuestionAnswer(
      state.aiQuestions,
      questionId,
      newAnswer
    );
    onStateChange({ aiQuestions: updatedQuestions });
  };

  // Follow-up questions mutation with loading state
  const followUpMutation = useMutation({
    mutationFn: async () => {
      // Store current questions as answered
      const newAnsweredQuestions = [...state.answeredQuestions, ...state.aiQuestions];
      const newRound = state.questionRound + 1;
      
      onStateChange({
        answeredQuestions: newAnsweredQuestions,
        questionRound: newRound
      });

      return await assessmentApi.followUpQuestions(
        state.selectedImage,
        state.audience,
        state.model,
        state.aiQuestions,
        state.woundClassification,
        newRound
      );
    },
    onSuccess: (data) => {
      // Update classification with revised confidence
      if (data.updatedClassification) {
        onStateChange({ 
          woundClassification: data.updatedClassification
        });
      }
      
      // Show confidence improvement
      if (data.updatedConfidence) {
        const oldConfidence = state.woundClassification?.confidence || 0.5;
        const newConfidence = data.updatedConfidence;
        if (newConfidence > oldConfidence) {
          toast({
            title: "Confidence Improved",
            description: `Assessment confidence increased to ${Math.round(newConfidence * 100)}%`
          });
        }
      }
      
      if (data.shouldProceedToPlan) {
        // Confidence reached 80%+, proceed to care plan generation
        onStateChange({ 
          aiQuestions: [],
          currentStep: 'generating-plan'
        });
        onNextStep();
        
        toast({
          title: "Ready for Care Plan",
          description: `Confidence reached ${Math.round(data.updatedConfidence * 100)}%. Generating care plan...`
        });
      } else if (data.needsMoreQuestions && data.questions && data.questions.length > 0) {
        // Need more questions to reach 80% confidence
        onStateChange({ aiQuestions: data.questions });
        toast({
          title: "Additional Questions Generated",
          description: `Confidence: ${Math.round(data.updatedConfidence * 100)}%. Need to reach 80% for final assessment.`
        });
      } else {
        // Confidence reached but no more questions - proceed anyway
        onStateChange({ 
          aiQuestions: [],
          currentStep: 'generating-plan'
        });
        onNextStep();
        
        toast({
          title: "Questions Complete",
          description: "Proceeding to care plan generation."
        });
      }
    },
    onError: (error: any) => {
      console.error('Follow-up questions error:', error);
      toast({
        title: "Error",
        description: "Failed to generate follow-up questions. Please try again.",
        variant: "destructive"
      });
    }
  });

  // Handle follow-up questions
  const handleFollowUpQuestions = () => {
    // Show immediate feedback
    toast({
      title: "Processing Answers",
      description: "Analyzing your responses to generate follow-up questions...",
    });
    
    followUpMutation.mutate();
  };

  const handleProceedToPlan = () => {
    onStateChange({ currentStep: 'generating-plan' });
    onNextStep();
  };

  return (
    <div className="space-y-6">
      {/* Thumbnail Image Display */}
      {state.selectedImage && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Camera className="h-5 w-5 mr-2 text-medical-blue" />
              Assessment Image
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex justify-center">
              <div className="bg-white border-2 border-gray-200 rounded-lg p-2 shadow-sm">
                <img 
                  src={URL.createObjectURL(state.selectedImage)} 
                  alt="Wound assessment"
                  className="h-32 w-32 object-cover rounded-lg"
                />
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Step 3: Diagnostic Questions</CardTitle>
          
          {/* Detection Information Only - No Duplicate Image */}
          {state.woundClassification?.detectionMetadata && (
            <div className="mb-4 p-3 bg-gray-50 rounded-lg border text-sm">
              <div className="font-medium text-gray-700 mb-1">Analysis Methods Used</div>
              <div className="space-y-1 text-xs text-gray-600">
                <div><strong>Detection:</strong> {getDetectionMethodName(state.woundClassification.detectionMetadata.model)}</div>
                <div><strong>Classification:</strong> {state.woundClassification.classificationMethod || 'AI Vision'}</div>
              </div>
            </div>
          )}
          
          <p className="text-gray-600">
            {state.aiQuestions.length > 0 ? (
              "The AI needs more information to improve its diagnosis. Please answer these questions:"
            ) : (
              "The AI is confident in its initial assessment. Proceeding to care plan generation."
            )}
          </p>
          {state.aiQuestions.length > 0 && (
            <div className="mt-4 bg-blue-50 p-4 rounded-lg border border-blue-200">
              <p className="text-sm text-blue-800">
                <strong>Important:</strong> More detailed answers will result in better assessment accuracy. 
                Please provide as much relevant information as possible for each question.
              </p>
            </div>
          )}
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
                <Badge variant={state.woundClassification.confidence > 0.80 ? 'default' : 'secondary'}>
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
            
            {/* Image Upload for Photo-Related Questions */}
            {(question.question.toLowerCase().includes('photo') || 
              question.question.toLowerCase().includes('image') || 
              question.question.toLowerCase().includes('picture')) && (
              <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <Label className="text-sm font-medium text-blue-800">
                  <Upload className="h-4 w-4 inline mr-2" />
                  Upload Additional Photo (Optional)
                </Label>
                <div className="mt-2">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) {
                        toast({
                          title: "Image Upload",
                          description: "Additional image functionality will be enhanced in the next iteration."
                        });
                      }
                    }}
                    className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                  />
                </div>
                <p className="text-xs text-blue-600 mt-1">
                  Upload a clearer photo, different angle, or close-up view to help with assessment
                </p>
              </div>
            )}
            
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
                  disabled={followUpMutation.isPending}
                  className="w-full bg-yellow-600 hover:bg-yellow-700 disabled:opacity-75"
                >
                  {followUpMutation.isPending ? (
                    <>
                      <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                      Processing Answers...
                    </>
                  ) : (
                    <>
                      Submit Answers to Improve Assessment
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </>
                  )}
                </Button>
              )}
              

            </div>
          ) : (
            <Button 
              onClick={handleProceedToPlan}
              className="w-full bg-medical-blue hover:bg-medical-blue/90 mt-4"
            >
              Generate Final Care Plan
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          )}
        </CardContent>
      </Card>
    </div>
  );
} 