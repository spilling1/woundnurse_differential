import { useLocation, useSearch } from "wouter";
import { ArrowLeft, ClipboardList, AlertTriangle, ThumbsUp, ThumbsDown, Download, Printer, UserCheck, Calendar, MapPin, User, FileText, Trash2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { useMutation, useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/useAuth";
import { useState, useRef } from "react";

export default function CarePlan() {
  const [, setLocation] = useLocation();
  const searchParams = useSearch();
  const { toast } = useToast();
  const { isAuthenticated } = useAuth();
  const [feedbackText, setFeedbackText] = useState("");
  const printRef = useRef<HTMLDivElement>(null);

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

  const handleDownloadPDF = () => {
    if (printRef.current) {
      const printWindow = window.open('', '_blank');
      if (printWindow) {
        printWindow.document.write(`
          <html>
            <head>
              <title>Wound Care Assessment - Case ${caseId}</title>
              <style>
                body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                .header { text-align: center; border-bottom: 2px solid #2563eb; padding-bottom: 20px; margin-bottom: 30px; }
                .section { margin-bottom: 30px; page-break-inside: avoid; }
                .section-title { color: #2563eb; font-size: 18px; font-weight: bold; margin-bottom: 15px; border-bottom: 1px solid #e5e7eb; padding-bottom: 5px; }
                .wound-images { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
                .wound-image { text-align: center; }
                .wound-image img { max-width: 200px; max-height: 200px; border: 1px solid #d1d5db; border-radius: 8px; }
                .assessment-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                .assessment-item { padding: 10px; background: #f9fafb; border-radius: 6px; }
                .assessment-label { font-weight: bold; color: #374151; }
                .care-plan { background: #f8fafc; padding: 20px; border-radius: 8px; border-left: 4px solid #2563eb; }
                .disclaimer { background: #fef3c7; padding: 15px; border-radius: 6px; border: 1px solid #f59e0b; margin: 20px 0; }
                .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e5e7eb; font-size: 12px; color: #6b7280; }
                @media print { .no-print { display: none; } }
              </style>
            </head>
            <body>
              ${printRef.current.innerHTML}
            </body>
          </html>
        `);
        printWindow.document.close();
        printWindow.print();
      }
    }
  };

  const handlePrint = () => {
    handleDownloadPDF();
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
                onClick={() => setLocation(isAuthenticated ? '/my-cases' : '/')}
                className="mr-4"
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                {isAuthenticated ? 'Back to Cases' : 'Back to Assessment'}
              </Button>
              <div className="flex items-center">
                <ClipboardList className="text-medical-blue text-xl mr-3" />
                <h1 className="text-xl font-semibold text-gray-900">Care Plan</h1>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <Badge variant="secondary">Case: {caseId}</Badge>
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => setLocation(`/nurse-evaluation?caseId=${caseId}`)}
              >
                <UserCheck className="mr-2 h-4 w-4" />
                Nurse Review
              </Button>
              <Button variant="outline" size="sm" onClick={handleDownloadPDF}>
                <Download className="mr-2 h-4 w-4" />
                Download PDF
              </Button>
              <Button variant="outline" size="sm" onClick={handlePrint}>
                <Printer className="mr-2 h-4 w-4" />
                Print
              </Button>
            </div>
          </div>
        </div>
      </nav>

      {/* Printable Content */}
      <div ref={printRef} className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* Professional Header */}
        <div className="header mb-8 text-center border-b-2 border-medical-blue pb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-3">Wound Care Assessment Report</h1>
          <div className="flex justify-center items-center space-x-6 text-sm text-gray-600">
            <div className="flex items-center">
              <Calendar className="h-4 w-4 mr-1" />
              {new Date().toLocaleDateString()}
            </div>
            <div className="flex items-center">
              <FileText className="h-4 w-4 mr-1" />
              Case ID: {caseId}
            </div>
            <div className="flex items-center">
              <User className="h-4 w-4 mr-1" />
              AI Model: {assessmentData.model}
            </div>
          </div>
        </div>

        {/* Wound Images Section */}
        {assessmentData.imageData && (
          <Card className="mb-6">
            <CardHeader>
              <CardTitle className="flex items-center">
                <MapPin className="h-5 w-5 mr-2 text-medical-blue" />
                Wound Documentation
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div className="text-center">
                  <img 
                    src={`data:${assessmentData.imageMimeType};base64,${assessmentData.imageData}`} 
                    alt="Wound assessment image"
                    className="w-full h-48 object-cover rounded-lg border border-gray-200 shadow-sm"
                  />
                  <p className="text-sm text-gray-600 mt-2">Assessment Image</p>
                  <p className="text-xs text-gray-500">
                    {(assessmentData.imageSize / 1024).toFixed(1)} KB • {assessmentData.imageMimeType}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Clinical Assessment */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center">
              <ClipboardList className="h-5 w-5 mr-2 text-medical-blue" />
              Clinical Assessment
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="font-semibold text-gray-700 mb-1">Wound Type</div>
                <div className="text-gray-900">{assessmentData.classification?.woundType || 'Not specified'}</div>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="font-semibold text-gray-700 mb-1">Wound Stage</div>
                <div className="text-gray-900">{assessmentData.classification?.stage || 'Not applicable'}</div>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="font-semibold text-gray-700 mb-1">Size Assessment</div>
                <div className="text-gray-900 capitalize">{assessmentData.classification?.size || 'Not specified'}</div>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="font-semibold text-gray-700 mb-1">Anatomical Location</div>
                <div className="text-gray-900">{assessmentData.classification?.location || 'Not specified'}</div>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="font-semibold text-gray-700 mb-1">Exudate Level</div>
                <div className="text-gray-900">{assessmentData.classification?.exudate || 'Not assessed'}</div>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="font-semibold text-gray-700 mb-1">Tissue Type</div>
                <div className="text-gray-900">{assessmentData.classification?.tissueType || 'Not specified'}</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Patient Context */}
        {assessmentData.contextData && (
          <Card className="mb-6">
            <CardHeader>
              <CardTitle className="flex items-center">
                <User className="h-5 w-5 mr-2 text-medical-blue" />
                Patient Context & History
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                {Object.entries(assessmentData.contextData).map(([key, value]) => {
                  if (!value) return null;
                  const labels: Record<string, string> = {
                    age: 'Age',
                    woundSite: 'Wound Site',
                    woundOrigin: 'Wound Origin',
                    comorbidities: 'Comorbidities',
                    medications: 'Current Medications',
                    nutritionStatus: 'Nutritional Status',
                    mobilityStatus: 'Mobility Status',
                    smokingStatus: 'Smoking Status',
                    alcoholUse: 'Alcohol Use',
                    stressLevel: 'Stress Level'
                  };
                  return (
                    <div key={key} className="bg-gray-50 p-3 rounded">
                      <div className="font-medium text-gray-700 mb-1">{labels[key] || key}</div>
                      <div className="text-gray-600">{value}</div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Evidence-Based Care Plan */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center">
              <FileText className="h-5 w-5 mr-2 text-medical-blue" />
              Evidence-Based Care Plan
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-blue-50 border-l-4 border-medical-blue p-6 rounded-lg">
              <div className="prose max-w-none">
                {formatCarePlan(assessmentData.carePlan)}
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
                  <p className="text-sm text-yellow-800 font-medium mb-2">Important Medical Disclaimer</p>
                  <p className="text-sm text-yellow-700">
                    This is an AI-generated care plan based on image analysis and provided context. This assessment is for educational and informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional before implementing any wound care recommendations.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Professional Footer */}
        <div className="text-center text-sm text-gray-500 mt-8 pt-6 border-t border-gray-200">
          <p className="font-medium">Generated by Wound Nurses AI Assessment System</p>
          <p>Report Date: {new Date().toLocaleDateString()} | Case ID: {caseId}</p>
          <p className="text-xs mt-2">This report contains confidential medical information</p>
        </div>
      </div>

      {/* Non-printable Feedback Section */}
      <div className="no-print max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 pb-8">

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