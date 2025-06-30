import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { useMutation } from "@tanstack/react-query";
import { Camera, Upload, CheckCircle, AlertCircle, RefreshCw, ArrowRight, Edit, ThumbsUp, ThumbsDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

interface AIGeneratedQuestion {
  id: string;
  question: string;
  answer: string;
  category: string;
  confidence: number;
}

interface WoundClassification {
  woundType: string;
  confidence: number;
  alternativeTypes: Array<{
    type: string;
    confidence: number;
    reasoning: string;
  }>;
}

interface PreliminaryCareplan {
  assessment: string;
  recommendations: string[];
  confidence: number;
  needsMoreInfo: boolean;
  additionalQuestions?: string[];
}

type FlowStep = 'audience' | 'upload' | 'ai-questions' | 'preliminary-plan' | 'final-plan';

export default function NewAssessmentFlow() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  
  // Flow state
  const [currentStep, setCurrentStep] = useState<FlowStep>('audience');
  const [audience, setAudience] = useState<'family' | 'patient' | 'medical'>('family');
  const [model, setModel] = useState<'gpt-4o' | 'gpt-3.5' | 'gpt-3.5-pro' | 'gemini-2.5-flash' | 'gemini-2.5-pro'>('gpt-4o');
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  
  // AI-generated data
  const [aiQuestions, setAiQuestions] = useState<AIGeneratedQuestion[]>([]);
  const [woundClassification, setWoundClassification] = useState<WoundClassification | null>(null);
  const [preliminaryPlan, setPreliminaryPlan] = useState<PreliminaryCareplan | null>(null);
  const [finalCaseId, setFinalCaseId] = useState<string | null>(null);
  
  // User input
  const [userFeedback, setUserFeedback] = useState<string>('');
  const [selectedAlternative, setSelectedAlternative] = useState<string | null>(null);
  const [questionRound, setQuestionRound] = useState<number>(1);
  const [answeredQuestions, setAnsweredQuestions] = useState<AIGeneratedQuestion[]>([]);

  // File upload handler
  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  // Step 1: Initial image analysis mutation
  const initialAnalysisMutation = useMutation({
    mutationFn: async () => {
      if (!selectedImage) throw new Error('No image selected');
      
      const formData = new FormData();
      formData.append('image', selectedImage);
      formData.append('audience', audience);
      formData.append('model', model);
      formData.append('analysisType', 'initial');
      
      const response = await apiRequest('POST', '/api/assessment/initial-analysis', formData);
      return await response.json();
    },
    onSuccess: (data: any) => {
      setAiQuestions(data.questions || []);
      setWoundClassification(data.classification);
      setCurrentStep('ai-questions');
    },
    onError: (error: any) => {
      toast({
        title: "Analysis Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  // Step 2: Generate preliminary care plan
  const preliminaryPlanMutation = useMutation({
    mutationFn: async () => {
      const updatedQuestions = aiQuestions.map(q => ({
        ...q,
        answer: q.answer
      }));
      
      const response = await apiRequest('POST', '/api/assessment/preliminary-plan', {
        imageData: selectedImage,
        audience,
        model,
        questions: updatedQuestions,
        classification: woundClassification,
        selectedAlternative,
        userFeedback
      });
      return await response.json();
    },
    onSuccess: (data: any) => {
      setPreliminaryPlan(data);
      setCurrentStep('preliminary-plan');
    },
    onError: (error: any) => {
      toast({
        title: "Plan Generation Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  // Auto-trigger preliminary plan when reaching that step with no pending questions
  useEffect(() => {
    if (currentStep === 'preliminary-plan' && !preliminaryPlan && aiQuestions.length === 0 && !preliminaryPlanMutation.isPending) {
      console.log('Auto-triggering preliminary plan generation');
      preliminaryPlanMutation.mutate();
    }
  }, [currentStep, preliminaryPlan, aiQuestions.length, preliminaryPlanMutation]);

  // Step 3: Generate final care plan
  const finalPlanMutation = useMutation({
    mutationFn: async () => {
      const formData = new FormData();
      if (selectedImage) {
        formData.append('image', selectedImage);
      }
      formData.append('audience', audience);
      formData.append('model', model);
      formData.append('questions', JSON.stringify(aiQuestions));
      formData.append('classification', JSON.stringify(woundClassification));
      formData.append('preliminaryPlan', JSON.stringify(preliminaryPlan));
      formData.append('userFeedback', userFeedback);

      const response = await fetch('/api/assessment/final-plan', {
        method: 'POST',
        body: formData,
        credentials: 'include',
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    },
    onSuccess: (data: any) => {
      setFinalCaseId(data.caseId);
      setCurrentStep('final-plan');
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

  // Update AI answer
  const updateAnswer = (questionId: string, newAnswer: string) => {
    setAiQuestions(prev => 
      prev.map(q => 
        q.id === questionId ? { ...q, answer: newAnswer } : q
      )
    );
  };

  const handleFeedbackQuestions = async () => {
    try {
      const response = await apiRequest('POST', '/api/assessment/feedback-questions', {
        classification: woundClassification,
        userFeedback,
        audience,
        model
      });
      
      const data = await response.json();
      
      if (data.questions && data.questions.length > 0) {
        setAiQuestions(data.questions);
        setCurrentStep('ai-questions');
        toast({
          title: "Follow-up Questions Generated",
          description: "Additional questions based on your feedback."
        });
      } else {
        toast({
          title: "No Additional Questions",
          description: "Your feedback is clear. Proceeding with current assessment."
        });
      }
    } catch (error) {
      console.error('Feedback questions error:', error);
      toast({
        title: "Error",
        description: "Failed to generate questions from feedback. Please try again.",
        variant: "destructive"
      });
    }
  };

  const handleFollowUpQuestions = async () => {
    try {
      // Store current questions as answered
      setAnsweredQuestions(prev => [...prev, ...aiQuestions]);
      
      // Increment round
      setQuestionRound(prev => prev + 1);
      
      // Generate new questions based on previous answers
      const formData = new FormData();
      if (selectedImage) {
        formData.append('image', selectedImage);
      }
      formData.append('audience', audience);
      formData.append('model', model);
      formData.append('previousQuestions', JSON.stringify(aiQuestions));
      formData.append('classification', JSON.stringify(woundClassification));
      formData.append('round', (questionRound + 1).toString());

      const response = await fetch('/api/assessment/follow-up-questions', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Failed to generate follow-up questions');
      }

      const data = await response.json();
      
      if (data.needsMoreQuestions && data.questions && data.questions.length > 0) {
        setAiQuestions(data.questions);
        toast({
          title: "Follow-up Questions Generated",
          description: `Round ${data.round} questions based on your Agent Instructions.`
        });
      } else {
        // No more questions needed, proceed to care plan
        setAiQuestions([]);
        setCurrentStep('preliminary-plan');
        
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

  // Render step content
  const renderStepContent = () => {
    switch (currentStep) {
      case 'audience':
        return (
          <Card>
            <CardHeader>
              <CardTitle>Step 1: Select Your Audience</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-gray-600">Who will be using this care plan?</p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {[
                  { value: 'family', label: 'Family Caregiver', desc: 'Simple, step-by-step guidance' },
                  { value: 'patient', label: 'Patient', desc: 'Self-care focused instructions' },
                  { value: 'medical', label: 'Medical Professional', desc: 'Clinical terminology and protocols' }
                ].map(option => (
                  <div
                    key={option.value}
                    className={`p-4 border rounded-lg cursor-pointer transition-all ${
                      audience === option.value 
                        ? 'border-medical-blue bg-blue-50' 
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => setAudience(option.value as any)}
                  >
                    <div className="font-medium">{option.label}</div>
                    <div className="text-sm text-gray-600">{option.desc}</div>
                  </div>
                ))}
              </div>
              
              <div className="mt-6">
                <Label>AI Model</Label>
                <select 
                  value={model} 
                  onChange={(e) => setModel(e.target.value as any)}
                  className="w-full mt-1 p-2 border rounded-md"
                >
                  <option value="gpt-4o">GPT-4o (Recommended)</option>
                  <option value="gpt-3.5">GPT-3.5</option>
                  <option value="gpt-3.5-pro">GPT-3.5 Pro</option>
                  <option value="gemini-2.5-flash">Gemini 2.5 Flash</option>
                  <option value="gemini-2.5-pro">Gemini 2.5 Pro</option>
                </select>
              </div>
              
              <Button 
                onClick={() => setCurrentStep('upload')}
                className="w-full bg-medical-blue hover:bg-medical-blue/90"
              >
                Continue to Image Upload
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </CardContent>
          </Card>
        );

      case 'upload':
        return (
          <Card>
            <CardHeader>
              <CardTitle>Step 2: Upload Wound Image</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                {imagePreview ? (
                  <div className="space-y-4">
                    <img 
                      src={imagePreview} 
                      alt="Wound preview" 
                      className="max-w-full h-64 object-contain mx-auto rounded-lg"
                    />
                    <p className="text-sm text-gray-600">Click below to change image</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <Camera className="h-12 w-12 text-gray-400 mx-auto" />
                    <p className="text-gray-600">Click to upload a wound image</p>
                  </div>
                )}
                
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageSelect}
                  className="hidden"
                  id="image-upload"
                />
                <label htmlFor="image-upload">
                  <Button variant="outline" className="mt-4" asChild>
                    <span>
                      <Upload className="mr-2 h-4 w-4" />
                      {imagePreview ? 'Change Image' : 'Upload Image'}
                    </span>
                  </Button>
                </label>
              </div>
              
              {selectedImage && (
                <Button 
                  onClick={() => initialAnalysisMutation.mutate()}
                  disabled={initialAnalysisMutation.isPending}
                  className="w-full bg-medical-blue hover:bg-medical-blue/90"
                >
                  {initialAnalysisMutation.isPending ? (
                    <>
                      <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing Image...
                    </>
                  ) : (
                    <>
                      Start AI Analysis
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </>
                  )}
                </Button>
              )}
            </CardContent>
          </Card>
        );

      case 'ai-questions':
        return (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Step 3: Diagnostic Questions</CardTitle>
                <p className="text-gray-600">
                  {aiQuestions.length > 0 ? (
                    "The AI needs more information to improve its diagnosis. Please answer these questions:"
                  ) : (
                    "The AI is confident in its initial assessment. Proceeding to care plan generation."
                  )}
                </p>
              </CardHeader>
            </Card>

            {woundClassification && (
              <Card>
                <CardHeader>
                  <CardTitle>Initial Classification</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <div className="font-medium">{woundClassification.woundType}</div>
                      <Badge variant={woundClassification.confidence > 0.75 ? 'default' : 'secondary'}>
                        {Math.round(woundClassification.confidence * 100)}% confidence
                      </Badge>
                    </div>
                  </div>
                  
                  {woundClassification?.alternativeTypes?.length > 0 && (
                    <div className="mt-4">
                      <Label>Alternative Classifications:</Label>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-2">
                        {woundClassification?.alternativeTypes?.map((alt, index) => (
                          <div
                            key={index}
                            className={`p-3 border rounded cursor-pointer transition-all ${
                              selectedAlternative === alt.type 
                                ? 'border-medical-blue bg-blue-50' 
                                : 'border-gray-200 hover:border-gray-300'
                            }`}
                            onClick={() => setSelectedAlternative(alt.type)}
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

            {aiQuestions.map((question) => (
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
                  value={userFeedback}
                  onChange={(e) => setUserFeedback(e.target.value)}
                  placeholder="Provide any additional information about the wound or patient that might help..."
                  rows={3}
                  className="mt-2"
                />
                
                {aiQuestions.length > 0 ? (
                  <div className="space-y-2">
                    {questionRound < 3 && (
                      <Button 
                        onClick={handleFollowUpQuestions}
                        className="w-full bg-yellow-600 hover:bg-yellow-700"
                      >
                        Check for Follow-up Questions (Round {questionRound + 1})
                        <ArrowRight className="ml-2 h-4 w-4" />
                      </Button>
                    )}
                    <Button 
                      onClick={() => preliminaryPlanMutation.mutate()}
                      disabled={preliminaryPlanMutation.isPending}
                      className="w-full bg-medical-blue hover:bg-medical-blue/90"
                    >
                      {preliminaryPlanMutation.isPending ? (
                        <>
                          <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                          Generating Care Plan...
                        </>
                      ) : (
                        <>
                          Generate Care Plan
                          <ArrowRight className="ml-2 h-4 w-4" />
                        </>
                      )}
                    </Button>
                  </div>
                ) : (
                  <Button 
                    onClick={() => preliminaryPlanMutation.mutate()}
                    disabled={preliminaryPlanMutation.isPending}
                    className="w-full bg-medical-blue hover:bg-medical-blue/90"
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

      case 'preliminary-plan':
        return (
          <div className="space-y-6">
            {preliminaryPlanMutation.isPending ? (
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
            ) : !preliminaryPlan ? (
              <Card>
                <CardHeader>
                  <CardTitle>Step 4: Preliminary Care Plan</CardTitle>
                  <p className="text-gray-600">
                    Preparing your preliminary assessment...
                  </p>
                </CardHeader>
              </Card>
            ) : (
              <>
                <Card className="border-l-4 border-l-medical-blue shadow-md">
                  <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50">
                    <CardTitle className="flex items-center justify-between">
                      <div className="flex items-center">
                        <CheckCircle className="h-6 w-6 text-medical-blue mr-3" />
                        Preliminary Assessment Complete
                      </div>
                      <Badge variant={preliminaryPlan.confidence > 0.75 ? "default" : "secondary"} className="bg-medical-blue">
                        {Math.round(preliminaryPlan.confidence * 100)}% Confidence
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
                        <div className="text-gray-700 leading-relaxed whitespace-pre-wrap">{preliminaryPlan.assessment}</div>
                      </div>
                    </div>
                    
                    {preliminaryPlan.recommendations && preliminaryPlan.recommendations.length > 0 && (
                      <div className="bg-green-50 rounded-lg p-4 border border-green-200 mt-4">
                        <h4 className="font-semibold text-gray-900 mb-3 flex items-center">
                          <CheckCircle className="h-5 w-5 text-green-600 mr-2" />
                          Key Recommendations
                        </h4>
                        <ul className="space-y-3">
                          {preliminaryPlan.recommendations.map((rec, index) => (
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

                <Card>
                  <CardHeader>
                    <CardTitle>Preliminary Recommendations</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-2">
                      {preliminaryPlan.recommendations.map((rec, index) => (
                        <li key={index} className="flex items-start">
                          <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                          <span>{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>

                {preliminaryPlan.needsMoreInfo && preliminaryPlan.additionalQuestions && (
                  <Card className="border-amber-200 bg-amber-50/30">
                    <CardHeader>
                      <CardTitle className="flex items-center">
                        <AlertCircle className="h-5 w-5 text-amber-500 mr-2" />
                        Additional Questions Needed
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-gray-600 mb-4">
                        {userFeedback ? 
                          "Based on your feedback, the AI needs clarification on these points:" :
                          "The AI needs more information to provide a confident assessment. Please answer these additional questions:"
                        }
                      </p>
                      <div className="space-y-4">
                        {preliminaryPlan.additionalQuestions.map((question, index) => (
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
                          onClick={() => {
                            // Collect answers and regenerate plan
                            preliminaryPlanMutation.mutate();
                          }}
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
                      value={userFeedback}
                      onChange={(e) => setUserFeedback(e.target.value)}
                      placeholder="Add corrections, additional context, or specific concerns about the preliminary assessment..."
                      rows={4}
                      className="border-amber-200 focus:border-amber-400 bg-white"
                    />
                    
                    <div className="flex gap-4 mt-4">
                      <Button 
                        onClick={() => preliminaryPlanMutation.mutate()}
                        disabled={preliminaryPlanMutation.isPending}
                        variant="outline"
                        className="flex-1"
                      >
                        <RefreshCw className="mr-2 h-4 w-4" />
                        Regenerate Plan
                      </Button>
                      

                      
                      {preliminaryPlan.confidence > 0.75 && (
                        <Button 
                          onClick={() => finalPlanMutation.mutate()}
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
              </>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Progress indicator */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          {['audience', 'upload', 'ai-questions', 'preliminary-plan', 'final-plan'].map((step, index) => {
            const stepLabels = ['Audience', 'Upload', 'AI Analysis', 'Preliminary Plan', 'Final Plan'];
            const isActive = currentStep === step;
            const isCompleted = ['audience', 'upload', 'ai-questions', 'preliminary-plan', 'final-plan'].indexOf(currentStep) > index;
            
            return (
              <div key={step} className="flex items-center">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                  isCompleted ? 'bg-green-500 text-white' :
                  isActive ? 'bg-medical-blue text-white' : 
                  'bg-gray-200 text-gray-600'
                }`}>
                  {isCompleted ? <CheckCircle className="h-4 w-4" /> : index + 1}
                </div>
                <div className="ml-2 text-sm font-medium text-gray-600">
                  {stepLabels[index]}
                </div>
                {index < 4 && (
                  <div className={`w-16 h-1 mx-4 ${
                    isCompleted ? 'bg-green-500' : 'bg-gray-200'
                  }`} />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {renderStepContent()}
    </div>
  );
}