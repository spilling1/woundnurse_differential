import { useLocation, useSearch } from "wouter";
import { ArrowLeft, ClipboardList, AlertTriangle, ThumbsUp, ThumbsDown, Download, Print } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { useMutation, useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { useState } from "react";

export default function CarePlan() {
  const [, setLocation] = useLocation();
  const searchParams = useSearch();
  const { toast } = useToast();
  const [feedbackText, setFeedbackText] = useState("");

  // Extract case ID from URL params
  const caseId = new URLSearchParams(searchParams).get('caseId');

  const { data: assessmentData, isLoading } = useQuery({
    queryKey: ['/api/assessment', caseId],
    enabled: !!caseId,
    queryFn: () => fetch(`/api/assessment/${caseId}`).then(res => res.json()),
  });

  const feedbackMutation = useMutation({
    mutationFn: async ({ feedbackType, comments }: { feedbackType: string; comments?: string }) => {
      return apiRequest('POST', '/api/feedback', {
        caseId,
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
    if (!caseId) return;
    
    feedbackMutation.mutate({
      feedbackType,
      comments: feedbackText || undefined
    });
  };

  const formatCarePlan = (plan: string) => {
    if (!plan) return null;
    
    const sections = plan.split('\n\n');
    return sections.map((section, index) => {
      if (section.includes('MEDICAL DISCLAIMER')) {
        return null;
      }
      return (
        <div key={index} className="mb-4">
          <p className="text-gray-700 whitespace-pre-wrap leading-relaxed">{section}</p>
        </div>
      );
    });
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-bg-light flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-medical-blue mx-auto mb-3"></div>
          <p className="text-gray-600">Loading care plan...</p>
        </div>
      </div>
    );
  }

  if (!assessmentData) {
    return (
      <div className="min-h-screen bg-bg-light flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-600 mb-4">Care plan not found</p>
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
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <Button 
                variant="ghost" 
                onClick={() => setLocation('/')}
                className="mr-4"
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Assessment
              </Button>
              <div className="flex items-center">
                <ClipboardList className="text-medical-blue text-xl mr-3" />
                <h1 className="text-xl font-semibold text-gray-900">Care Plan</h1>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <Badge variant="secondary">Case: {caseId}</Badge>
              <Button variant="outline" size="sm">
                <Download className="mr-2 h-4 w-4" />
                Download
              </Button>
              <Button variant="outline" size="sm">
                <Print className="mr-2 h-4 w-4" />
                Print
              </Button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Assessment Summary */}
        <Card className="mb-6">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900">Assessment Summary</h2>
              <Badge variant="secondary">{assessmentData.model}</Badge>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <span className="font-medium text-gray-700">Wound Type:</span>
                <span className="ml-2 text-gray-600">{assessmentData.classification?.woundType}</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">Stage:</span>
                <span className="ml-2 text-gray-600">{assessmentData.classification?.stage}</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">Size:</span>
                <span className="ml-2 text-gray-600 capitalize">{assessmentData.classification?.size}</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">Location:</span>
                <span className="ml-2 text-gray-600">{assessmentData.classification?.location}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Medical Disclaimer */}
        <Card className="mb-6">
          <CardContent className="p-6">
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <div className="flex items-start">
                <AlertTriangle className="text-yellow-600 mr-3 mt-1 h-5 w-5" />
                <div>
                  <p className="text-sm text-yellow-800 font-medium">Important Medical Disclaimer</p>
                  <p className="text-sm text-yellow-700 mt-1">
                    This is an AI-generated care plan based on image analysis. Please consult a healthcare professional before following any recommendations. This tool is for educational purposes and should not replace professional medical advice.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Care Plan Content */}
        <Card className="mb-6">
          <CardContent className="p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Personalized Care Plan</h2>
            <div className="prose max-w-none">
              {formatCarePlan(assessmentData.plan)}
            </div>
          </CardContent>
        </Card>

        {/* Feedback Section */}
        <Card>
          <CardContent className="p-6">
            <h3 className="font-semibold text-gray-900 mb-4">Feedback</h3>
            <p className="text-sm text-gray-600 mb-4">
              Your feedback helps us improve the accuracy and usefulness of our AI-generated care plans.
            </p>
            
            <div className="space-y-4">
              <div className="flex space-x-3">
                <Button 
                  className="flex-1 bg-success-green hover:bg-green-700"
                  onClick={() => handleFeedback('helpful')}
                  disabled={feedbackMutation.isPending}
                >
                  <ThumbsUp className="mr-2 h-4 w-4" />
                  This plan was helpful
                </Button>
                <Button 
                  className="flex-1 bg-alert-red hover:bg-red-700"
                  onClick={() => handleFeedback('not-helpful')}
                  disabled={feedbackMutation.isPending}
                >
                  <ThumbsDown className="mr-2 h-4 w-4" />
                  This plan needs improvement
                </Button>
              </div>
              
              <Textarea
                value={feedbackText}
                onChange={(e) => setFeedbackText(e.target.value)}
                rows={4}
                placeholder="Please share specific feedback about the care plan - what was helpful, what could be improved, or any additional context that might help us generate better recommendations..."
                className="text-sm"
              />
              
              <Button 
                onClick={() => handleFeedback(feedbackText ? 'helpful' : 'not-helpful')}
                disabled={feedbackMutation.isPending}
                className="w-full"
              >
                {feedbackMutation.isPending ? 'Submitting...' : 'Submit Detailed Feedback'}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}