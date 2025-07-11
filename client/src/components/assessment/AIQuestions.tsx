import { CheckCircle, ArrowRight, RefreshCw, Camera, Upload, Info, X } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";
import type { StepProps } from "./shared/AssessmentTypes";
import { assessmentApi, assessmentHelpers } from "./shared/AssessmentUtils";
import DetectionTransparencyCard from "./DetectionTransparencyCard";
import DifferentialDiagnosisQuestions from "./DifferentialDiagnosisQuestions";



export default function AIQuestions({ state, onStateChange, onNextStep }: StepProps) {
  const { toast } = useToast();
  
  // Handle additional image uploads
  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
      toast({
        title: "Invalid file type",
        description: "Please upload an image file (JPG, PNG, GIF, etc.)",
        variant: "destructive",
      });
      return;
    }
    
    // Validate file size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      toast({
        title: "File too large",
        description: "Please upload an image smaller than 10MB",
        variant: "destructive",
      });
      return;
    }
    
    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      const preview = e.target?.result as string;
      const imageId = `img_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      // Add to selectedImages array
      const newImage = {
        file,
        preview,
        id: imageId,
        description: ''
      };
      
      const updatedImages = [...state.selectedImages, newImage];
      onStateChange({ 
        selectedImages: updatedImages,
        selectedImage: file, // Keep compatibility
        imagePreview: preview
      });
      
      toast({
        title: "Image added",
        description: "Additional image uploaded successfully",
      });
    };
    reader.readAsDataURL(file);
  };
  
  // Remove additional image
  const removeAdditionalImage = (imageId: string) => {
    const updatedImages = state.selectedImages.filter(img => img.id !== imageId);
    onStateChange({ selectedImages: updatedImages });
    
    toast({
      title: "Image removed",
      description: "Additional image removed successfully",
    });
  };
  
  // Calculate potential confidence improvement
  const calculateConfidenceImprovement = (): number => {
    const currentConfidence = state.woundClassification?.confidence || 0;
    if (currentConfidence >= 0.9) return 0;
    
    // Base improvement from additional images
    const imageImprovement = Math.min(15, (5 - state.selectedImages.length) * 3);
    return Math.max(0, imageImprovement);
  };

  // Helper function to get user-friendly detection method names
  const getDetectionMethodName = (model: string): string => {
    switch (model) {
      case 'yolo8':
      case 'yolov8':
      case 'smart-yolo-yolo':
        return 'YOLO v8 Detection';
      case 'smart-yolo-color':
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

  // Helper function to calculate confidence improvement based on question category
  const getConfidenceImprovement = (category: string): number => {
    switch (category) {
      case 'diagnostic':
        return 8; // High impact on diagnosis accuracy
      case 'treatment':
        return 5; // Medium impact on treatment planning
      case 'medical_history':
        return 6; // Important for context
      case 'location':
        return 7; // Critical for wound assessment
      case 'symptoms':
        return 4; // Helpful for care planning
      case 'general':
        return 3; // Lower impact but still useful
      default:
        return 5; // Default moderate improvement
    }
  };

  // Helper function to get improvement type description
  const getImprovementType = (category: string): string => {
    switch (category) {
      case 'diagnostic':
        return 'Confidence Improvement';
      case 'location':
        return 'Confidence Improvement';
      case 'medical_history':
        return 'Confidence Improvement';
      case 'treatment':
        return 'Care Plan Improvement';
      case 'symptoms':
        return 'Care Plan Improvement';
      case 'general':
        return 'Care Plan Improvement';
      default:
        return 'Assessment Quality';
    }
  };

  // Helper function to get priority level description
  const getPriorityLevel = (category: string): string => {
    switch (category) {
      case 'diagnostic':
        return 'High Priority';
      case 'location':
        return 'High Priority';
      case 'medical_history':
        return 'Medium Priority';
      case 'treatment':
        return 'Medium Priority';
      case 'symptoms':
        return 'Low Priority';
      case 'general':
        return 'Low Priority';
      default:
        return 'Medium Priority';
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
        newRound,
        state.bodyRegion
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
        // CRITICAL FIX: Preserve current round's answered questions before proceeding
        const currentRoundAnswered = state.aiQuestions.filter(q => q.answer && q.answer.trim() !== '');
        const allAnsweredQuestions = [...state.answeredQuestions, ...currentRoundAnswered];
        
        onStateChange({ 
          aiQuestions: [],
          currentStep: 'generating-plan',
          answeredQuestions: allAnsweredQuestions
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
        // CRITICAL FIX: Preserve current round's answered questions before proceeding
        const currentRoundAnswered = state.aiQuestions.filter(q => q.answer && q.answer.trim() !== '');
        const allAnsweredQuestions = [...state.answeredQuestions, ...currentRoundAnswered];
        
        onStateChange({ 
          aiQuestions: [],
          currentStep: 'generating-plan',
          answeredQuestions: allAnsweredQuestions
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
    // CRITICAL FIX: Preserve ALL answered questions from all rounds for the care plan generation
    const currentRoundAnswered = state.aiQuestions.filter(q => q.answer && q.answer.trim() !== '');
    const allAnsweredQuestions = [...state.answeredQuestions, ...currentRoundAnswered];
    
    console.log(`Preserving ${allAnsweredQuestions.length} total answered questions for care plan generation`);
    console.log('All answered questions:', allAnsweredQuestions.map(q => ({ question: q.question, answer: q.answer })));
    
    onStateChange({ 
      currentStep: 'generating-plan',
      answeredQuestions: allAnsweredQuestions
    });
    onNextStep();
  };

  return (
    <div className="space-y-6">
      {/* Thumbnail Image Display */}
      {state.selectedImages && state.selectedImages.length > 0 && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Camera className="h-5 w-5 mr-2 text-medical-blue" />
              Assessment Images ({state.selectedImages.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {state.selectedImages.map((image, index) => (
                <div key={image.id || index} className="bg-white border-2 border-gray-200 rounded-lg p-2 shadow-sm">
                  <img 
                    src={image.preview || URL.createObjectURL(image.file)} 
                    alt={`Wound assessment ${index + 1}`}
                    className="w-full h-32 object-contain rounded"
                  />
                  <p className="text-xs text-gray-500 mt-1 text-center">
                    Image {index + 1} {index === 0 ? '(Primary)' : ''}
                  </p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Confidence Indicator */}
      {state.woundClassification?.confidence && state.woundClassification.confidence < 0.9 && (
        <Alert className="mb-4 bg-blue-50 border-blue-200">
          <Info className="h-4 w-4 text-blue-600" />
          <AlertDescription className="text-blue-800">
            <strong>Current assessment confidence: {Math.round(state.woundClassification.confidence * 100)}%</strong>
            <br />
            Adding additional images could improve confidence by up to {calculateConfidenceImprovement()}%.
          </AlertDescription>
        </Alert>
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
          
          {/* Multiple Wounds Warning */}
          {state.woundClassification?.multipleWounds && (
            <div className="mb-4 p-3 bg-yellow-50 rounded-lg border border-yellow-200 text-sm">
              <div className="font-medium text-yellow-800 mb-1">Multiple Wounds Detected</div>
              <div className="text-yellow-700">
                The AI has identified that your images may show different wounds. This assessment focuses on the primary wound.
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
            <CardTitle className="text-xl font-bold">Most Probable Diagnoses</CardTitle>
          </CardHeader>
          <CardContent>
            {/* Primary Diagnosis with enhanced styling */}
            <div className="mb-6">
              <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <div className="text-lg font-bold text-blue-900">
                    1. {state.woundClassification.woundType}
                  </div>
                  <Badge variant="default" className="bg-blue-600 text-white">
                    {/* Use the normalized confidence from differential diagnosis if available */}
                    {state.woundClassification?.differentialDiagnosis?.possibleTypes?.[0]?.confidence 
                      ? Math.round(state.woundClassification.differentialDiagnosis.possibleTypes[0].confidence * 100)
                      : Math.round(state.woundClassification.confidence * 100)}% confidence
                  </Badge>
                </div>
                {state.woundClassification.stage && (
                  <div className="text-sm text-blue-700 font-medium">
                    ({state.woundClassification.stage})
                  </div>
                )}
              </div>
              
              {/* AI Reasoning */}
              {state.woundClassification?.reasoning && (
                <div className="mt-3 p-3 bg-gray-50 rounded-lg border">
                  <div className="text-sm font-medium text-gray-700 mb-2">AI Analysis Reasoning:</div>
                  <div className="text-sm text-gray-600 leading-relaxed">
                    {state.woundClassification.reasoning}
                  </div>
                </div>
              )}
            </div>

            {/* Differential Diagnosis Section - Only show secondary possibilities */}
            {state.woundClassification?.differentialDiagnosis?.possibleTypes ? (
              <div className="mt-6">
                <h3 className="text-lg font-bold mb-4">Differential Diagnosis</h3>
                <div className="space-y-3">
                  {state.woundClassification.differentialDiagnosis.possibleTypes
                    .filter((possibility, index) => index > 0) // Skip the first (primary) diagnosis
                    .map((possibility, index) => (
                    <div key={index} className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <div className="text-base font-semibold text-gray-900">
                          {index + 2}. {possibility.woundType}
                        </div>
                        <Badge variant="outline" className="text-gray-700 border-gray-300">
                          {Math.round(possibility.confidence * 100)}% confidence
                        </Badge>
                      </div>
                      <div className="text-sm text-gray-600 leading-relaxed">
                        {possibility.reasoning}
                      </div>
                    </div>
                  ))}
                </div>
                
                {/* Interactive Differential Diagnosis Questions */}
                {state.woundClassification.differentialDiagnosis.questionsToAsk && 
                 state.woundClassification.differentialDiagnosis.questionsToAsk.length > 0 && (
                  <DifferentialDiagnosisQuestions 
                    questions={state.woundClassification.differentialDiagnosis.questionsToAsk}
                    originalClassification={state.woundClassification}
                    audience={state.audience}
                    model={state.model}
                    onRefinementComplete={(refinement) => {
                      // Update state with Page 2 analysis
                      onStateChange({
                        differentialRefinement: refinement,
                        showPage2Analysis: true
                      });
                      
                      // If user wants to proceed to care plan generation
                      if (refinement.proceedToCarePlan) {
                        onNextStep();
                      }
                    }}
                  />
                )}
              </div>
            ) : (
              /* Fallback when no differential diagnosis is provided */
              <div className="mt-6 p-4 bg-orange-50 border border-orange-200 rounded-lg">
                <div className="text-orange-900 font-medium mb-2">
                  Confidence in Each Diagnosis
                </div>
                <div className="text-sm text-orange-800">
                  Based on visual assessment alone, it is difficult to assign precise confidence scores. 
                  Additional clinical information would be needed to gain extreme confidence in the primary diagnosis.
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Enhanced Detection Transparency */}
      {state.woundClassification && (
        <DetectionTransparencyCard 
          classification={state.woundClassification}
        />
      )}

      {/* Questions Section */}
      {state.aiQuestions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Follow-up Questions</CardTitle>
            <p className="text-sm text-gray-600 mt-2">
              These questions will help improve diagnostic accuracy and provide better care recommendations.
            </p>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
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
                            onChange={handleImageUpload}
                            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                          />
                        </div>
                        <p className="text-xs text-blue-600 mt-1">
                          Upload a clearer photo, different angle, or close-up view to help with assessment
                        </p>
                      </div>
                    )}
                    
                    <div className="flex items-center justify-between mt-2">
                      <div className="flex items-center text-sm text-blue-600">
                        <ArrowRight className="h-4 w-4 mr-1" />
                        {getImprovementType(question.category)}: +{getConfidenceImprovement(question.category)}%
                      </div>
                      <div className="text-xs text-gray-500">
                        {getPriorityLevel(question.category)}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Additional Image Upload Section */}
      {state.woundClassification?.confidence && state.woundClassification.confidence < 0.9 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Camera className="h-5 w-5 mr-2 text-medical-blue" />
              Upload Additional Images
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <p className="text-sm text-gray-600">
                Adding more images can help improve assessment accuracy. Consider uploading:
              </p>
              <ul className="text-sm text-gray-600 ml-4 list-disc space-y-1">
                <li>Different angles or lighting</li>
                <li>Close-up views of wound edges</li>
                <li>Photos with size reference (coin, ruler)</li>
                <li>Wider context showing surrounding skin</li>
              </ul>
              
              <div className="mt-4">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-medical-blue file:text-white hover:file:bg-medical-blue/90"
                />
              </div>
              
              {/* Display Additional Images */}
              {state.selectedImages.length > 1 && (
                <div className="mt-4">
                  <Label className="text-sm font-medium text-gray-700">Additional Images ({state.selectedImages.length - 1})</Label>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mt-2">
                    {state.selectedImages.slice(1).map((image, index) => (
                      <div key={image.id} className="relative bg-white border-2 border-gray-200 rounded-lg p-2 shadow-sm">
                        <img 
                          src={image.preview} 
                          alt={`Additional view ${index + 1}`}
                          className="w-full h-24 object-cover rounded-lg"
                        />
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => removeAdditionalImage(image.id)}
                          className="absolute top-1 right-1 h-6 w-6 p-0 bg-white/90 border-gray-300 text-red-600 hover:bg-red-50"
                        >
                          <X className="h-3 w-3" />
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

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
              {state.questionRound < 3 ? (
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
              ) : (
                <Button 
                  onClick={handleProceedToPlan}
                  className="w-full bg-medical-blue hover:bg-medical-blue/90"
                >
                  Generate Care Plan with Current Information
                  <ArrowRight className="ml-2 h-4 w-4" />
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