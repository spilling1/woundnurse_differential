import { ClipboardList, AlertTriangle, ThumbsUp, ThumbsDown, ExternalLink } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { useState } from "react";
import { useLocation } from "wouter";

interface CarePlanSectionProps {
  assessmentData: any;
  model: string;
  isProcessing: boolean;
}

export default function CarePlanSection({ assessmentData, model, isProcessing }: CarePlanSectionProps) {
  const [, setLocation] = useLocation();
  const [feedbackText, setFeedbackText] = useState("");
  const { toast } = useToast();

  const feedbackMutation = useMutation({
    mutationFn: async ({ feedbackType, comments }: { feedbackType: string; comments?: string }) => {
      return apiRequest('POST', '/api/feedback', {
        caseId: assessmentData.caseId,
        feedbackType,
        comments
      });
    },
    onSuccess: () => {
      toast({
        title: "Feedback Submitted",
        description: "Thank you for your feedback. It helps improve our AI model.",
      });
      setFeedbackText("");
    },
    onError: (error: any) => {
      toast({
        title: "Feedback Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const handleFeedback = (feedbackType: 'helpful' | 'not-helpful') => {
    if (!assessmentData?.caseId) return;
    
    feedbackMutation.mutate({
      feedbackType,
      comments: feedbackText || undefined
    });
  };

  const formatCarePlan = (plan: string) => {
    if (!plan) return null;
    
    // Split the plan into sections for better formatting
    const sections = plan.split('\n\n');
    return sections.map((section, index) => {
      if (section.includes('MEDICAL DISCLAIMER')) {
        return null; // We'll handle this separately
      }
      return (
        <div key={index} className="mb-4">
          <p className="text-gray-700 text-sm whitespace-pre-wrap">{section}</p>
        </div>
      );
    });
  };

  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex items-center mb-6">
          <ClipboardList className="text-medical-blue text-lg mr-2" />
          <h2 className="text-lg font-semibold text-gray-900">Care Plan</h2>
          {assessmentData && (
            <Badge variant="secondary" className="ml-auto">
              {model}
            </Badge>
          )}
        </div>

        <div className="space-y-6">
          {/* Medical Disclaimer */}
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <div className="flex items-start">
              <AlertTriangle className="text-yellow-600 mr-2 mt-1 h-4 w-4" />
              <div>
                <p className="text-sm text-yellow-800 font-medium">Medical Disclaimer</p>
                <p className="text-sm text-yellow-700 mt-1">
                  This is an AI-generated plan. Please consult a healthcare professional before following recommendations.
                </p>
              </div>
            </div>
          </div>

          {/* Care Instructions */}
          {assessmentData && (
            <div className="space-y-4">
              <div className="flex justify-between items-center mb-4">
                <p className="text-sm text-gray-600">
                  Preview of your personalized care plan
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setLocation(`/care-plan?caseId=${assessmentData.caseId}`)}
                >
                  <ExternalLink className="mr-2 h-4 w-4" />
                  View Full Care Plan
                </Button>
              </div>
              {formatCarePlan(assessmentData.plan)}

              {/* Feedback Section */}
              <div className="mt-8 pt-6 border-t border-gray-200">
                <h3 className="font-semibold text-gray-900 mb-3">Feedback</h3>
                <div className="space-y-3">
                  <div className="flex space-x-2">
                    <Button 
                      className="flex-1 bg-success-green hover:bg-green-700"
                      onClick={() => handleFeedback('helpful')}
                      disabled={feedbackMutation.isPending}
                    >
                      <ThumbsUp className="mr-2 h-4 w-4" />
                      Helpful
                    </Button>
                    <Button 
                      className="flex-1 bg-alert-red hover:bg-red-700"
                      onClick={() => handleFeedback('not-helpful')}
                      disabled={feedbackMutation.isPending}
                    >
                      <ThumbsDown className="mr-2 h-4 w-4" />
                      Not Helpful
                    </Button>
                  </div>
                  <Textarea
                    value={feedbackText}
                    onChange={(e) => setFeedbackText(e.target.value)}
                    rows={3}
                    placeholder="Additional comments or suggestions (optional)..."
                    className="text-sm"
                  />
                  <Button 
                    variant="secondary"
                    onClick={() => handleFeedback(feedbackText ? 'helpful' : 'not-helpful')}
                    disabled={feedbackMutation.isPending}
                  >
                    {feedbackMutation.isPending ? 'Submitting...' : 'Submit Feedback'}
                  </Button>
                </div>
              </div>
            </div>
          )}

          {/* Loading State */}
          {!assessmentData && !isProcessing && (
            <div className="text-center py-8">
              <ClipboardList className="mx-auto h-12 w-12 text-gray-400 mb-3" />
              <p className="text-gray-600">Complete the configuration and upload an image to generate your care plan</p>
            </div>
          )}

          {/* Processing State */}
          {isProcessing && (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-medical-blue mx-auto mb-3"></div>
              <p className="text-gray-600">Generating personalized care plan...</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
