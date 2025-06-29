import { useLocation, useParams, useSearch } from "wouter";
import { ArrowLeft, ClipboardList, AlertTriangle, ThumbsUp, ThumbsDown, Download, Printer, UserCheck, Calendar, MapPin, User, FileText, Plus, LogOut, Settings } from "lucide-react";
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
import type { WoundAssessment } from "@shared/schema";

export default function CarePlan() {
  const [, setLocation] = useLocation();
  const params = useParams();
  const search = useSearch();
  const { toast } = useToast();
  const { isAuthenticated } = useAuth();
  const [feedbackText, setFeedbackText] = useState("");
  const printRef = useRef<HTMLDivElement>(null);

  // Extract case ID from URL params with multiple fallback methods
  let caseId = params.caseId;
  
  if (!caseId || caseId === 'care-plan') {
    // Fallback: extract from window.location.pathname
    const pathParts = window.location.pathname.split('/');
    const caseIdIndex = pathParts.indexOf('care-plan') + 1;
    if (caseIdIndex > 0 && caseIdIndex < pathParts.length) {
      caseId = pathParts[caseIdIndex];
    }
  }
  
  if (!caseId || caseId === 'care-plan') {
    // Fallback: extract from query parameters
    const urlParams = new URLSearchParams(window.location.search);
    const queryCaseId = urlParams.get('caseId');
    if (queryCaseId) {
      caseId = queryCaseId;
    }
  }
  
  if (!caseId || caseId === 'care-plan') {
    // Final fallback: extract from hash if present
    const hash = window.location.hash.replace('#', '');
    if (hash && hash !== 'care-plan') {
      caseId = hash;
    }
  }
  
  // Extract version from query parameters
  const urlParams = new URLSearchParams(search);
  const requestedVersion = urlParams.get('version');

  const { data: assessmentData, isLoading, error } = useQuery<WoundAssessment>({
    queryKey: requestedVersion ? [`/api/assessment/${caseId}?version=${requestedVersion}`] : [`/api/assessment/${caseId}`],
    enabled: !!caseId,
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
              <meta charset="UTF-8">
              <style>
                /* Reset and Base Styles */
                * { box-sizing: border-box; margin: 0; padding: 0; }
                body { 
                  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                  font-size: 11pt;
                  line-height: 1.4;
                  color: #1f2937;
                  background: white;
                  margin: 0;
                  -webkit-print-color-adjust: exact;
                  print-color-adjust: exact;
                }
                
                /* Page Setup */
                @page {
                  size: A4;
                  margin: 1in;
                  @top-center {
                    content: "Wound Care Assessment Report";
                    font-size: 9pt;
                    color: #6b7280;
                  }
                  @bottom-center {
                    content: "Page " counter(page) " of " counter(pages);
                    font-size: 9pt;
                    color: #6b7280;
                  }
                }
                
                /* Typography */
                h1 { font-size: 24pt; font-weight: 700; color: #1f2937; margin-bottom: 8pt; }
                h2 { font-size: 16pt; font-weight: 600; color: #2563eb; margin: 20pt 0 12pt 0; }
                h3 { font-size: 14pt; font-weight: 600; color: #374151; margin: 16pt 0 8pt 0; }
                h4 { font-size: 12pt; font-weight: 600; color: #4b5563; margin: 12pt 0 6pt 0; }
                p { margin-bottom: 8pt; }
                
                /* Header Styles */
                .header {
                  text-align: center;
                  border-bottom: 3pt solid #2563eb;
                  padding-bottom: 20pt;
                  margin-bottom: 30pt;
                  page-break-after: avoid;
                }
                
                .header h1 {
                  margin-bottom: 12pt;
                  color: #1f2937;
                }
                
                .header-info {
                  display: flex;
                  justify-content: center;
                  gap: 30pt;
                  font-size: 10pt;
                  color: #6b7280;
                  margin-top: 10pt;
                }
                
                .header-info-item {
                  display: flex;
                  align-items: center;
                  gap: 4pt;
                }
                
                /* Card Styles */
                .card {
                  border: 1pt solid #e5e7eb;
                  border-radius: 8pt;
                  margin-bottom: 20pt;
                  page-break-inside: avoid;
                  background: white;
                  box-shadow: 0 1pt 3pt rgba(0,0,0,0.1);
                }
                
                .card-header {
                  background: #f8fafc;
                  border-bottom: 1pt solid #e5e7eb;
                  padding: 16pt;
                  border-radius: 8pt 8pt 0 0;
                }
                
                .card-title {
                  font-size: 14pt;
                  font-weight: 600;
                  color: #1f2937;
                  display: flex;
                  align-items: center;
                  gap: 8pt;
                }
                
                .card-content {
                  padding: 16pt;
                }
                
                /* Grid Layouts */
                .assessment-grid {
                  display: grid;
                  grid-template-columns: 1fr 1fr;
                  gap: 12pt;
                  margin: 16pt 0;
                }
                
                .assessment-item {
                  background: #f9fafb;
                  border: 1pt solid #e5e7eb;
                  border-radius: 6pt;
                  padding: 12pt;
                }
                
                .assessment-label {
                  font-weight: 600;
                  color: #374151;
                  font-size: 10pt;
                  margin-bottom: 4pt;
                  text-transform: uppercase;
                  letter-spacing: 0.025em;
                }
                
                .assessment-value {
                  color: #1f2937;
                  font-size: 11pt;
                  line-height: 1.3;
                }
                
                /* Context Grid */
                .context-grid {
                  display: grid;
                  grid-template-columns: 1fr 1fr;
                  gap: 12pt;
                  margin: 16pt 0;
                }
                
                .context-item {
                  background: #f9fafb;
                  border-left: 3pt solid #2563eb;
                  padding: 10pt;
                  border-radius: 0 4pt 4pt 0;
                }
                
                .context-label {
                  font-weight: 600;
                  color: #374151;
                  font-size: 10pt;
                  margin-bottom: 4pt;
                }
                
                .context-value {
                  color: #4b5563;
                  font-size: 10pt;
                  line-height: 1.3;
                }
                
                /* Image Styles */
                .wound-images {
                  display: grid;
                  grid-template-columns: repeat(auto-fit, minmax(200pt, 1fr));
                  gap: 16pt;
                  margin: 16pt 0;
                }
                
                .wound-image {
                  text-align: center;
                  border: 1pt solid #d1d5db;
                  border-radius: 8pt;
                  padding: 12pt;
                  background: white;
                }
                
                .wound-image img {
                  max-width: 100%;
                  max-height: 200pt;
                  border-radius: 4pt;
                  box-shadow: 0 2pt 4pt rgba(0,0,0,0.1);
                }
                
                .image-caption {
                  font-size: 9pt;
                  color: #6b7280;
                  margin-top: 8pt;
                  font-style: italic;
                }
                
                /* Care Plan Styles */
                .care-plan {
                  background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                  border: 1pt solid #0ea5e9;
                  border-left: 4pt solid #2563eb;
                  border-radius: 8pt;
                  padding: 20pt;
                  margin: 16pt 0;
                }
                
                .care-plan h3 {
                  color: #1e40af;
                  margin-bottom: 12pt;
                }
                
                .care-plan p {
                  margin-bottom: 10pt;
                  line-height: 1.5;
                }
                
                .care-plan ul {
                  margin: 10pt 0 10pt 20pt;
                }
                
                .care-plan li {
                  margin-bottom: 6pt;
                  line-height: 1.4;
                }
                
                /* Disclaimer Styles */
                .disclaimer {
                  background: #fef3c7;
                  border: 2pt solid #f59e0b;
                  border-radius: 8pt;
                  padding: 16pt;
                  margin: 20pt 0;
                  display: flex;
                  align-items: flex-start;
                  gap: 12pt;
                }
                
                .disclaimer-icon {
                  color: #d97706;
                  font-weight: bold;
                  font-size: 16pt;
                }
                
                .disclaimer-content {
                  flex: 1;
                }
                
                .disclaimer-title {
                  font-weight: 600;
                  color: #92400e;
                  font-size: 11pt;
                  margin-bottom: 8pt;
                }
                
                .disclaimer-text {
                  color: #a16207;
                  font-size: 10pt;
                  line-height: 1.4;
                }
                
                /* Footer Styles */
                .footer {
                  text-align: center;
                  margin-top: 40pt;
                  padding-top: 20pt;
                  border-top: 1pt solid #e5e7eb;
                  font-size: 9pt;
                  color: #6b7280;
                  page-break-inside: avoid;
                }
                
                .footer-title {
                  font-weight: 600;
                  color: #374151;
                  margin-bottom: 4pt;
                }
                
                .footer-details {
                  margin-bottom: 8pt;
                }
                
                .footer-confidential {
                  font-size: 8pt;
                  color: #9ca3af;
                  font-style: italic;
                }
                
                /* Print Optimizations */
                @media print {
                  .no-print { display: none !important; }
                  
                  .card {
                    box-shadow: none;
                    border: 1pt solid #d1d5db;
                  }
                  
                  .page-break {
                    page-break-before: always;
                  }
                  
                  .avoid-break {
                    page-break-inside: avoid;
                  }
                  
                  /* Ensure colors print */
                  * {
                    -webkit-print-color-adjust: exact !important;
                    print-color-adjust: exact !important;
                  }
                }
                
                /* Utility Classes */
                .text-center { text-align: center; }
                .text-left { text-align: left; }
                .text-right { text-align: right; }
                .font-bold { font-weight: 600; }
                .font-semibold { font-weight: 500; }
                .text-sm { font-size: 10pt; }
                .text-xs { font-size: 9pt; }
                .mb-2 { margin-bottom: 8pt; }
                .mb-4 { margin-bottom: 16pt; }
                .mt-2 { margin-top: 8pt; }
                .mt-4 { margin-top: 16pt; }
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
      
      // Convert markdown links to clickable links
      const formatWithLinks = (text: string) => {
        const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
        const parts = [];
        let lastIndex = 0;
        let match;
        
        while ((match = linkRegex.exec(text)) !== null) {
          // Add text before the link
          if (match.index > lastIndex) {
            parts.push(text.slice(lastIndex, match.index));
          }
          
          // Add the clickable link
          parts.push(
            <a 
              key={match.index}
              href={match[2]} 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-blue-600 hover:text-blue-800 underline font-medium"
            >
              {match[1]}
            </a>
          );
          
          lastIndex = match.index + match[0].length;
        }
        
        // Add remaining text after the last link
        if (lastIndex < text.length) {
          parts.push(text.slice(lastIndex));
        }
        
        return parts.length > 0 ? parts : text;
      };
      
      return (
        <div key={index} className="mb-4">
          <p className="text-gray-700 whitespace-pre-wrap leading-relaxed">
            {formatWithLinks(section)}
          </p>
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
          <p className="text-gray-500 text-sm mb-4">Case ID: {caseId}</p>
          {error && <p className="text-red-500 text-sm mb-4">Error: {error.message}</p>}
          <Button onClick={() => setLocation('/my-cases')}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to My Cases
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
                <div></div>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              {/* Care Plan Actions */}
              <div className="flex items-center space-x-2 border-r border-gray-200 pr-3">
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

              {/* User Actions */}
              {isAuthenticated && (
                <div className="flex items-center space-x-2">
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => setLocation('/assessment')}
                    className="border-medical-blue text-medical-blue hover:bg-medical-blue hover:text-white"
                  >
                    <Plus className="mr-2 h-4 w-4" />
                    Start New Case
                  </Button>
                  <Button 
                    variant="ghost"
                    size="sm"
                    onClick={() => setLocation("/settings")}
                    className="p-2"
                    title="Settings"
                  >
                    <Settings className="h-4 w-4" />
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => window.location.href = "/api/logout"}
                  >
                    <LogOut className="mr-2 h-4 w-4" />
                    Log Out
                  </Button>
                </div>
              )}
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
            
          </div>
        </div>

        {/* Wound Images Section */}
        {(assessmentData as any)?.imageData && (
          <Card className="mb-6 card">
            <CardHeader className="card-header">
              <CardTitle className="card-title">
                <MapPin className="h-5 w-5 mr-2 text-medical-blue" />
                Wound Documentation
              </CardTitle>
            </CardHeader>
            <CardContent className="card-content">
              <div className="wound-images">
                <div className="wound-image">
                  <img 
                    src={`data:${(assessmentData as any).imageMimeType};base64,${(assessmentData as any).imageData}`} 
                    alt="Wound assessment image"
                    className="w-full h-48 object-cover rounded-lg border border-gray-200 shadow-sm"
                  />
                  <p className="image-caption">Assessment Image</p>
                  <p className="image-caption">
                    {(((assessmentData as any).imageSize || 0) / 1024).toFixed(1)} KB • {(assessmentData as any).imageMimeType}
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
            <div className="assessment-grid">
              <div className="assessment-item">
                <div className="assessment-label">Wound Type</div>
                <div className="assessment-value">{(assessmentData?.classification as any)?.woundType || 'Not specified'}</div>
              </div>
              <div className="assessment-item">
                <div className="assessment-label">Wound Stage</div>
                <div className="assessment-value">{(assessmentData?.classification as any)?.stage || 'Not applicable'}</div>
              </div>
              <div className="assessment-item">
                <div className="assessment-label">Size Assessment</div>
                <div className="assessment-value">{(assessmentData?.classification as any)?.size || 'Not specified'}</div>
              </div>
              <div className="assessment-item">
                <div className="assessment-label">Anatomical Location</div>
                <div className="assessment-value">{(assessmentData?.classification as any)?.location || 'Not specified'}</div>
              </div>
              <div className="assessment-item">
                <div className="assessment-label">Exudate Level</div>
                <div className="assessment-value">{(assessmentData?.classification as any)?.exudate || 'Not assessed'}</div>
              </div>
              <div className="assessment-item">
                <div className="assessment-label">Tissue Type</div>
                <div className="assessment-value">{(assessmentData?.classification as any)?.tissueType || 'Not specified'}</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Patient Context */}
        {assessmentData?.contextData && (
          <Card className="mb-6 card">
            <CardHeader className="card-header">
              <CardTitle className="card-title">
                <User className="h-5 w-5 mr-2 text-medical-blue" />
                Patient Context & History
              </CardTitle>
            </CardHeader>
            <CardContent className="card-content">
              <div className="context-grid">
                {Object.entries(assessmentData.contextData as Record<string, any>).map(([key, value]) => {
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
                    <div key={key} className="context-item">
                      <div className="context-label">{labels[key] || key}</div>
                      <div className="context-value">{value}</div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Evidence-Based Care Plan */}
        <Card className="mb-6 card">
          <CardHeader className="card-header">
            <CardTitle className="card-title">
              <FileText className="h-5 w-5 mr-2 text-medical-blue" />
              Evidence-Based Care Plan
            </CardTitle>
          </CardHeader>
          <CardContent className="card-content">
            <div className="care-plan">
              <div className="prose max-w-none">
                {formatCarePlan(assessmentData?.carePlan || '')}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Medical Disclaimer */}
        <Card className="mb-6 card">
          <CardContent className="card-content">
            <div className="disclaimer">
              <div className="disclaimer-icon">⚠</div>
              <div className="disclaimer-content">
                <p className="disclaimer-title">Important Medical Disclaimer</p>
                <p className="disclaimer-text">
                  This is an AI-generated care plan based on image analysis and provided context. This assessment is for educational and informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional before implementing any wound care recommendations.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Professional Footer */}
        <div className="footer">
          <p className="footer-title">Generated by Wound Nurses AI Assessment System</p>
          <p className="footer-details">Report Date: {new Date().toLocaleDateString()} | Case ID: {caseId}</p>
          <p className="footer-confidential">This report contains confidential medical information</p>
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
              
              
              
              
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}