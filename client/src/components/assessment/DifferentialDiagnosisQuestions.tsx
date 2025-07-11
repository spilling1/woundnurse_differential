import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { ChevronRight, Loader2, CheckCircle, TrendingUp, X, MessageCircle } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";

interface DifferentialDiagnosisQuestionsProps {
  questions: string[];
  originalClassification: any;
  audience: string;
  model: string;
  onRefinementComplete: (refinement: any) => void;
}

interface QuestionAnswer {
  question: string;
  answer: string;
  importance: 'critical' | 'high' | 'medium';
}

export default function DifferentialDiagnosisQuestions({
  questions,
  originalClassification,
  audience,
  model,
  onRefinementComplete
}: DifferentialDiagnosisQuestionsProps) {
  const [questionAnswers, setQuestionAnswers] = useState<QuestionAnswer[]>(
    questions.map((q, index) => ({
      question: q,
      answer: '',
      importance: index < 2 ? 'critical' : index < 4 ? 'high' : 'medium'
    }))
  );
  const [showPage2, setShowPage2] = useState(false);
  const [refinementResult, setRefinementResult] = useState<any>(null);
  const [otherInformation, setOtherInformation] = useState('');
  const { toast } = useToast();

  // Update answer for a specific question
  const updateAnswer = (index: number, answer: string) => {
    setQuestionAnswers(prev => 
      prev.map((qa, i) => i === index ? { ...qa, answer } : qa)
    );
  };

  // Mutation for refining differential diagnosis
  const refineMutation = useMutation({
    mutationFn: async () => {
      const answeredQuestions = questionAnswers.filter(qa => qa.answer.trim() !== '');
      
      if (answeredQuestions.length === 0) {
        throw new Error('Please answer at least one question to refine the diagnosis');
      }

      const response = await apiRequest("POST", "/api/assessment/refine-differential-diagnosis", {
        originalClassification,
        questionAnswers: answeredQuestions,
        otherInformation: otherInformation.trim(),
        model
      });
      
      return await response.json();
    },
    onSuccess: (data) => {
      setRefinementResult(data);
      setShowPage2(true);
      onRefinementComplete(data);
      
      toast({
        title: "Analysis Complete",
        description: `Refined diagnosis with ${data.page2Analysis.confidence * 100}% confidence`,
      });
    },
    onError: (error: any) => {
      toast({
        title: "Analysis Error",
        description: error.message || "Failed to refine differential diagnosis",
        variant: "destructive",
      });
    }
  });

  const handleRefineAnalysis = () => {
    refineMutation.mutate();
  };

  // Get importance badge styling
  const getImportanceBadge = (importance: string) => {
    switch (importance) {
      case 'critical':
        return <Badge variant="destructive" className="text-xs">Critical</Badge>;
      case 'high':
        return <Badge variant="secondary" className="text-xs bg-orange-100 text-orange-800">High</Badge>;
      default:
        return <Badge variant="outline" className="text-xs">Medium</Badge>;
    }
  };

  if (showPage2 && refinementResult) {
    const confidence = refinementResult.page2Analysis.confidence;
    const primaryDiagnosis = refinementResult.page2Analysis.primaryDiagnosis?.woundType || 
                           refinementResult.page2Analysis.remaining?.[0]?.woundType || 
                           'Unknown';
    const confidencePercent = Math.round(confidence * 100);
    
    // Check if we have high confidence (90-95%) for final diagnosis display
    const isHighConfidence = confidence >= 0.90;
    
    return (
      <div className="space-y-6">

        {/* Clean Refined Diagnosis Display */}
        <Card>
          <CardContent className="pt-6">
            {/* Simple Unified Diagnosis Display */}
            <div className="mb-6">
              <h3 className="text-lg font-bold text-green-900 mb-3">Refined Diagnosis</h3>
              <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <div className="text-lg font-bold text-green-900">
                    {primaryDiagnosis}
                  </div>
                  <Badge variant="default" className="bg-green-600 text-white">
                    {confidencePercent}% confidence
                  </Badge>
                </div>
                <div className="mt-4">
                  <h4 className="font-semibold text-green-900 mb-2">Clinical Reasoning:</h4>
                  <div className="text-sm text-green-800 leading-relaxed bg-white p-3 rounded border border-green-200">
                    {typeof refinementResult.page2Analysis.reasoning === 'string' ? 
                      refinementResult.page2Analysis.reasoning : 
                      'Analysis based on patient answers with clinical logic-based refinement'}
                  </div>
                </div>
              </div>
            </div>

            {/* Eliminated Possibilities */}
            {refinementResult.page2Analysis.eliminated.length > 0 && (
              <div className="mb-6">
                <h3 className="text-lg font-bold text-red-900 mb-3">Eliminated Possibilities</h3>
                <div className="space-y-2">
                  {refinementResult.page2Analysis.eliminated.map((eliminated, index) => (
                    <div key={index} className="flex items-center p-3 bg-red-50 border border-red-200 rounded-lg">
                      <X className="h-4 w-4 text-red-600 mr-2" />
                      <span className="text-sm text-red-800 line-through">{eliminated}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Only show remaining possibilities for lower confidence cases */}
            {!isHighConfidence && refinementResult.page2Analysis.remaining.length > 1 && (
              <div className="mb-6">
                <h3 className="text-lg font-bold text-gray-900 mb-3">Remaining Possibilities</h3>
                <div className="space-y-3">
                  {refinementResult.page2Analysis.remaining.map((possibility, index) => (
                    <div key={index} className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <div className="text-base font-semibold text-gray-900">
                          {index + 1}. {possibility.woundType}
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
              </div>
            )}

            {/* Follow-up Questions Section */}
            <div className="mb-6">
              <h3 className="text-lg font-bold text-blue-900 mb-3">Follow-up Questions to Shape Care Plan</h3>
              
              {/* Generate follow-up questions based on refined diagnosis */}
              <div className="space-y-4">
                <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                  <div className="text-sm font-medium text-gray-900 mb-2">
                    What treatments or dressings are currently being used on the wound?
                  </div>
                  <Textarea
                    placeholder="Describe current treatments..."
                    className="min-h-[60px] resize-none"
                  />
                </div>
                
                <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                  <div className="text-sm font-medium text-gray-900 mb-2">
                    Are there any signs of infection (increased redness, warmth, swelling, discharge)?
                  </div>
                  <Textarea
                    placeholder="Describe any signs of infection..."
                    className="min-h-[60px] resize-none"
                  />
                </div>
                
                <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                  <div className="text-sm font-medium text-gray-900 mb-2">
                    How long has this wound been present?
                  </div>
                  <Textarea
                    placeholder="Timeline of wound development..."
                    className="min-h-[60px] resize-none"
                  />
                </div>
                
                <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                  <div className="text-sm font-medium text-gray-900 mb-2">
                    Beyond diabetes, what other medical conditions does the patient have (circulation issues, foot deformities, previous amputations)?
                  </div>
                  <Textarea
                    placeholder="Other medical conditions beyond diabetes..."
                    className="min-h-[60px] resize-none"
                  />
                </div>
                
                <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                  <div className="text-sm font-medium text-gray-900 mb-2">
                    What shoes does the patient wear every day, and are there any foot deformities or calluses?
                  </div>
                  <Textarea
                    placeholder="Describe footwear and foot condition..."
                    className="min-h-[60px] resize-none"
                  />
                </div>
                
                {/* Other Information Section */}
                <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="text-sm font-medium text-blue-900 mb-2">
                    Any other pertinent information that may impact the diagnosis (e.g. Surgeries, Radiation, Lacerations, health conditions)?
                  </div>
                  <Textarea
                    placeholder="Please provide any additional information that may impact the diagnosis..."
                    className="min-h-[80px] resize-none"
                  />
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="mt-6 flex gap-3 justify-center">
              <Button 
                onClick={() => {
                  // Trigger the next step in the assessment flow
                  onRefinementComplete({
                    ...refinementResult,
                    proceedToCarePlan: true
                  });
                }}
                className="flex items-center"
              >
                <TrendingUp className="h-4 w-4 mr-2" />
                Generate Care Plan
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <Card className="mt-6">
      <CardHeader>
        <CardTitle className="text-brown-900">
          Key Questions to Gain Extreme Confidence in Primary Diagnosis
        </CardTitle>
        <p className="text-sm text-brown-700">
          Answer these questions in order of clinical importance to refine the differential diagnosis.
          Your answers will help eliminate possibilities and improve diagnostic accuracy.
        </p>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {questionAnswers.map((qa, index) => (
            <div key={index} className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <div className="flex items-center mb-2">
                    <span className="text-brown-700 font-medium mr-2">{index + 1}.</span>
                    {getImportanceBadge(qa.importance)}
                  </div>
                  <Label htmlFor={`question-${index}`} className="text-sm font-medium text-gray-900">
                    {qa.question}
                  </Label>
                </div>
              </div>
              
              <Textarea
                id={`question-${index}`}
                placeholder={`Your answer for question ${index + 1}...`}
                value={qa.answer}
                onChange={(e) => updateAnswer(index, e.target.value)}
                className="min-h-[80px] resize-none"
              />
              
              {qa.answer.trim() !== '' && (
                <div className="mt-2 text-xs text-green-600 flex items-center">
                  <CheckCircle className="h-3 w-3 mr-1" />
                  Answer provided
                </div>
              )}
            </div>
          ))}

          {/* Other Information Section */}
          <div className="border border-blue-200 rounded-lg p-4 bg-blue-50">
            <div className="flex items-start justify-between mb-3">
              <div className="flex-1">
                <div className="flex items-center mb-2">
                  <span className="text-blue-700 font-medium mr-2">{questionAnswers.length + 1}.</span>
                </div>
                <Label htmlFor="other-information" className="text-sm font-medium text-blue-900">
                  Any other pertinent information that may impact the diagnosis (e.g. Surgeries, Radiation, Lacerations, health conditions)?
                </Label>
              </div>
            </div>
            
            <Textarea
              id="other-information"
              placeholder="Please provide any additional information that may impact the diagnosis..."
              value={otherInformation}
              onChange={(e) => setOtherInformation(e.target.value)}
              className="min-h-[80px] resize-none"
            />
            
            {otherInformation.trim() !== '' && (
              <div className="mt-2 text-xs text-green-600 flex items-center">
                <CheckCircle className="h-3 w-3 mr-1" />
                Answer provided
              </div>
            )}
          </div>

          {/* Instructions */}
          <Alert>
            <AlertDescription>
              <strong>Instructions:</strong> Answer the questions in order of importance (Critical → High → Medium). 
              You can analyze with partial answers, but more complete responses will provide better diagnostic refinement.
            </AlertDescription>
          </Alert>

          {/* Action Buttons */}
          <div className="flex gap-3">
            <Button 
              onClick={handleRefineAnalysis}
              disabled={refineMutation.isPending || questionAnswers.every(qa => qa.answer.trim() === '')}
              className="flex items-center"
            >
              {refineMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <TrendingUp className="h-4 w-4 mr-2" />
                  Refine Analysis (Page 2)
                </>
              )}
            </Button>
            
            <div className="text-sm text-gray-600 flex items-center">
              {/* Include "Other Information" field in the count */}
              {questionAnswers.filter(qa => qa.answer.trim() !== '').length + (otherInformation.trim() !== '' ? 1 : 0)} / {questionAnswers.length + 1} questions answered
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}