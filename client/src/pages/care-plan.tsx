import { useLocation, useParams, useSearch } from "wouter";
import { ArrowLeft, ClipboardList, AlertTriangle, ThumbsUp, ThumbsDown, Download, Printer, UserCheck, Calendar, MapPin, User, FileText, Plus, LogOut, Settings, RefreshCw, MoreVertical, Edit3, Save, X } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/useAuth";
import { useState, useRef, useEffect } from "react";
import type { WoundAssessment } from "@shared/schema";
import WoundVisualization from "@/components/WoundVisualization";
import AdminNavigation from "@/components/shared/AdminNavigation";

export default function CarePlan() {
  const [, setLocation] = useLocation();
  const params = useParams();
  const search = useSearch();
  const { toast } = useToast();
  const { isAuthenticated, user } = useAuth();
  const [feedbackText, setFeedbackText] = useState("");
  const [isEditingCaseName, setIsEditingCaseName] = useState(false);
  const [editedCaseName, setEditedCaseName] = useState("");
  const printRef = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();

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

  const { data: assessmentData, isLoading, error, refetch } = useQuery<WoundAssessment>({
    queryKey: requestedVersion ? [`/api/assessment/${caseId}?version=${requestedVersion}`] : [`/api/assessment/${caseId}`],
    enabled: !!caseId,
  });

  // Refresh data when page loads
  useEffect(() => {
    if (caseId) {
      refetch();
    }
  }, [caseId, refetch]);

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

  const updateCaseNameMutation = useMutation({
    mutationFn: async (caseName: string) => {
      return apiRequest('PATCH', `/api/case/${caseId}/name`, {
        caseName
      });
    },
    onSuccess: () => {
      toast({
        title: "Case Name Updated",
        description: "Case name has been updated successfully.",
      });
      setIsEditingCaseName(false);
      refetch();
    },
    onError: (error: any) => {
      toast({
        title: "Update Failed",
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

  const handleEditCaseName = () => {
    setEditedCaseName(assessmentData?.caseName || "");
    setIsEditingCaseName(true);
  };

  const handleSaveCaseName = () => {
    if (editedCaseName.trim()) {
      updateCaseNameMutation.mutate(editedCaseName.trim());
    }
  };

  const handleCancelEditCaseName = () => {
    setIsEditingCaseName(false);
    setEditedCaseName("");
  };

  const formatCarePlan = (plan: string) => {
    if (!plan) return null;
    
    // Remove any JSON structure that appears at the beginning of the care plan
    let cleanPlan = plan;
    
    // Remove JSON objects that start with { and include target_audience, wound_assessment, etc.
    cleanPlan = cleanPlan.replace(/^\s*\{[\s\S]*?\}\s*(?:\n|$)/, '');
    
    // Remove any remaining JSON-like structures or artifacts
    cleanPlan = cleanPlan.replace(/^json\s*\{[\s\S]*?\}\s*(?:\n|$)/i, '');
    cleanPlan = cleanPlan.replace(/^```json[\s\S]*?```\s*(?:\n|$)/i, '');
    cleanPlan = cleanPlan.replace(/^```[\s\S]*?```\s*(?:\n|$)/i, '');
    
    // Remove any leading quotes or artifacts
    cleanPlan = cleanPlan.replace(/^["'][\s\S]*?["']\s*(?:\n|$)/, '');
    
    // Remove any lines that look like JSON properties (e.g., "request": { "target_audience": "medical"...)
    cleanPlan = cleanPlan.replace(/^[\s\S]*?"request"\s*:\s*\{[\s\S]*?\}\s*(?:\n|$)/i, '');
    
    // Remove any remaining JSON-like content that starts with property names
    cleanPlan = cleanPlan.replace(/^[\s\S]*?(?:"target_audience"|"wound_assessment"|"type"|"stage"|"size")[\s\S]*?\}\s*(?:\n|$)/i, '');
    
    // Remove any text that starts with "json {" (case insensitive)
    cleanPlan = cleanPlan.replace(/^[\s\S]*?json\s*\{[\s\S]*?\}\s*(?:\n|$)/i, '');
    
    // Remove any remaining curly braces at the start if they appear to be JSON remnants
    cleanPlan = cleanPlan.replace(/^\s*\{[^}]*\}\s*(?:\n|$)/, '');
    
    // Remove any lines that contain typical JSON property patterns
    cleanPlan = cleanPlan.replace(/^[\s\S]*?("[\w_]+"\s*:\s*"[^"]*"[\s\S]*?)+\s*(?:\n|$)/i, '');
    
    // Clean up any remaining text that looks like "I READ YOUR STUPID INSTRUCTIONS" or similar
    cleanPlan = cleanPlan.replace(/^[\s\S]*?"?\s*I\s+READ\s+YOUR\s+STUPID\s+INSTRUCTIONS[\s\S]*?(?:\n|$)/i, '');
    
    // Remove any remaining quotes and brackets at the start
    cleanPlan = cleanPlan.replace(/^["'\}\]\s]*/, '');
    
    // Trim any excessive whitespace
    cleanPlan = cleanPlan.trim();
    
    // Remove MEDICAL DISCLAIMER since we handle it separately
    cleanPlan = cleanPlan.replace(/\*\*MEDICAL DISCLAIMER:\*\*[\s\S]*?\n\n/, '');
    
    // Extract detection analysis section for separate rendering
    const detectionAnalysisMatch = cleanPlan.match(/\*\*DETECTION SYSTEM ANALYSIS:\*\*[\s\S]*$/);
    const detectionAnalysis = detectionAnalysisMatch ? detectionAnalysisMatch[0] : '';
    
    // Remove detection analysis from main content
    const planWithoutDetection = cleanPlan.replace(/---\s*\n\n\*\*DETECTION SYSTEM ANALYSIS:\*\*[\s\S]*$/, '');
    
    // Check if the content contains HTML tags
    const hasHtmlTags = /<[^>]*>/g.test(planWithoutDetection);
    
    if (hasHtmlTags) {
      // If content contains HTML, render it as HTML with proper sanitization
      return (
        <div 
          dangerouslySetInnerHTML={{ __html: planWithoutDetection }}
          className="prose prose-lg max-w-none [&_p]:mb-4 [&_h1]:text-2xl [&_h1]:font-bold [&_h1]:mb-4 [&_h2]:text-xl [&_h2]:font-semibold [&_h2]:mb-3 [&_h3]:text-lg [&_h3]:font-medium [&_h3]:mb-2 [&_ul]:mb-4 [&_ol]:mb-4 [&_li]:mb-2"
        />
      );
    }
    
    const sections = planWithoutDetection.split('\n\n');
    return sections.map((section, index) => {
      // Skip empty sections and lines with just dashes
      if (!section.trim() || section.trim() === '---' || section.includes('MEDICAL DISCLAIMER')) {
        return null;
      }
      
      // Enhanced formatting logic
      const formatSection = (text: string) => {
        // Handle bullet points and numbered lists
        const lines = text.split('\n');
        const formattedLines = lines.map((line, lineIndex) => {
          // Skip empty lines and dash separators
          if (!line.trim() || line.trim() === '---') {
            return null;
          }
          
          // Convert markdown links to clickable links
          const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
          const formatWithLinks = (lineText: string) => {
            const parts = [];
            let lastIndex = 0;
            let match;
            
            while ((match = linkRegex.exec(lineText)) !== null) {
              if (match.index > lastIndex) {
                parts.push(lineText.slice(lastIndex, match.index));
              }
              parts.push(
                <a 
                  key={`${index}-${lineIndex}-${match.index}`}
                  href={match[2]} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-medical-blue hover:text-blue-700 underline font-medium"
                >
                  {match[1]}
                </a>
              );
              lastIndex = match.index + match[0].length;
            }
            
            if (lastIndex < lineText.length) {
              parts.push(lineText.slice(lastIndex));
            }
            
            return parts.length > 0 ? parts : lineText;
          };
          
          // Format bullet points
          if (line.match(/^\s*[-*•]\s/)) {
            return (
              <li key={lineIndex} className="mb-2 text-gray-700 leading-relaxed">
                {formatWithLinks(line.replace(/^\s*[-*•]\s/, ''))}
              </li>
            );
          }
          
          // Format numbered lists
          if (line.match(/^\s*\d+\.\s/)) {
            return (
              <li key={lineIndex} className="mb-2 text-gray-700 leading-relaxed">
                {formatWithLinks(line.replace(/^\s*\d+\.\s/, ''))}
              </li>
            );
          }
          
          // Regular paragraph
          if (line.trim()) {
            return (
              <p key={lineIndex} className="mb-3 text-gray-700 leading-relaxed">
                {formatWithLinks(line)}
              </p>
            );
          }
          
          return null;
        }).filter(Boolean);
        
        // If this section contains lists, wrap them properly
        const hasListItems = formattedLines.some(line => 
          line?.props?.className?.includes('mb-2') && line?.type === 'li'
        );
        
        if (hasListItems) {
          return (
            <ul className="list-disc ml-6 space-y-1">
              {formattedLines}
            </ul>
          );
        }
        
        return formattedLines;
      };
      
      // Check if this section is a heading
      const isMainHeading = section.match(/^(##\s|#\s|\*\*.*\*\*)/);
      
      if (isMainHeading) {
        return (
          <div key={index} className="mb-6">
            <h3 className="text-xl font-semibold text-gray-900 mb-4 pb-2 border-b border-gray-200">
              {section.replace(/^(##\s|#\s|\*\*)/g, '').replace(/\*\*$/g, '')}
            </h3>
          </div>
        );
      }
      
      // Check if this section starts with a keyword that should be a subheading
      const isSubHeading = section.match(/^(Understanding|Daily Care|Self Care|Instructions|Recommendations|Assessment|Plan|Treatment|Wound Care|Care Routine)/i);
      
      if (isSubHeading) {
        const lines = section.split('\n');
        const heading = lines[0];
        const content = lines.slice(1).join('\n');
        
        return (
          <div key={index} className="mb-6">
            <h4 className="text-lg font-semibold text-gray-800 mb-3 text-medical-blue">
              {heading.replace(/^#+\s*/, '')}
            </h4>
            <div className="pl-4 border-l-2 border-gray-200">
              {formatSection(content)}
            </div>
          </div>
        );
      }
      
      return (
        <div key={index} className="mb-4">
          {formatSection(section)}
        </div>
      );
    }).filter(Boolean);
  };

  // Function to render detection analysis separately (admin only)
  const renderDetectionAnalysis = (plan: string) => {
    if (!plan) return null;
    
    // Only show to admin users
    if (!user || user.role !== 'admin') {
      return null;
    }
    
    const detectionAnalysisMatch = plan.match(/\*\*DETECTION SYSTEM ANALYSIS:\*\*[\s\S]*$/);
    if (!detectionAnalysisMatch) return null;
    
    const detectionAnalysis = detectionAnalysisMatch[0];
    
    return (
      <div className="text-xs text-gray-500 mt-6 pt-4 border-t border-gray-200">
        <div className="whitespace-pre-wrap">
          {detectionAnalysis.replace(/\*\*/g, '')}
        </div>
      </div>
    );
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
                  onClick={() => refetch()}
                  disabled={isLoading}
                >
                  <RefreshCw className={`mr-2 h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
                  Refresh
                </Button>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => setLocation(`/nurse-evaluation/${caseId}`)}
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
                    onClick={() => setLocation("/profile")}
                    className="p-2"
                    title="My Profile"
                  >
                    <User className="h-4 w-4" />
                  </Button>
                  {(user as any)?.role === 'admin' && (
                    <Button 
                      variant="ghost"
                      size="sm"
                      onClick={() => setLocation("/settings")}
                      className="p-2"
                      title="Settings"
                    >
                      <Settings className="h-4 w-4" />
                    </Button>
                  )}
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => {
                      localStorage.removeItem('auth_token');
                      setLocation('/');
                    }}
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

      {/* Admin Navigation */}
      <AdminNavigation />

      {/* Printable Content */}
      <div ref={printRef} className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* Professional Header */}
        <div className="header mb-8 text-center border-b-2 border-medical-blue pb-6">
          <div className="flex justify-center items-center gap-4 mb-3">
            <h1 className="text-3xl font-bold text-gray-900">
              {assessmentData?.caseName || "Wound Care Assessment Report"}
            </h1>
            {isAuthenticated && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                    <MoreVertical className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={handleEditCaseName}>
                    <Edit3 className="h-4 w-4 mr-2" />
                    Edit Case Name
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            )}
          </div>
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

        {/* Image Display Section - Clean Display Only */}
        {(assessmentData as any)?.imageData && (
          <Card className="mb-8 shadow-md">
            <CardHeader className="bg-slate-50 border-b">
              <CardTitle className="flex items-center text-lg">
                <MapPin className="h-5 w-5 mr-3 text-medical-blue" />
                Assessment Image
              </CardTitle>
            </CardHeader>
            <CardContent className="p-6">
              <div className="flex justify-center">
                <div className="bg-white border-2 border-gray-200 rounded-xl p-6 shadow-sm max-w-2xl w-full">
                  <img 
                    src={`data:${(assessmentData as any).imageMimeType};base64,${(assessmentData as any).imageData}`} 
                    alt="Wound assessment image"
                    className="w-full max-h-96 object-contain rounded-lg border border-gray-100 cursor-pointer hover:shadow-lg transition-shadow"
                    onClick={(e) => {
                      // Create modal overlay for full-size view
                      const modal = document.createElement('div');
                      modal.className = 'fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4';
                      modal.onclick = () => modal.remove();
                      
                      const img = document.createElement('img');
                      img.src = (e.target as HTMLImageElement).src;
                      img.className = 'max-w-full max-h-full object-contain rounded-lg';
                      img.onclick = (e) => e.stopPropagation();
                      
                      const closeBtn = document.createElement('button');
                      closeBtn.innerHTML = '×';
                      closeBtn.className = 'absolute top-4 right-4 text-white text-4xl font-bold bg-black bg-opacity-50 rounded-full w-12 h-12 flex items-center justify-center hover:bg-opacity-75';
                      closeBtn.onclick = () => modal.remove();
                      
                      modal.appendChild(img);
                      modal.appendChild(closeBtn);
                      document.body.appendChild(modal);
                    }}
                  />
                  <div className="mt-4 text-center">
                    <p className="text-sm font-medium text-gray-700 mb-1">Assessment Image</p>
                    <p className="text-xs text-gray-500 mb-2">
                      {(((assessmentData as any).imageSize || 0) / 1024).toFixed(1)} KB • {(assessmentData as any).imageMimeType}
                    </p>
                    <p className="text-xs text-gray-400 mb-2">
                      Captured: {new Date(assessmentData?.createdAt || '').toLocaleDateString()}
                    </p>
                    <p className="text-xs text-blue-600 italic">
                      Click image to view full size for detailed medical review
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}



        {/* Clinical Assessment */}
        <Card className="mb-8 shadow-md">
          <CardHeader className="bg-slate-50 border-b">
            <CardTitle className="flex items-center text-lg">
              <ClipboardList className="h-5 w-5 mr-3 text-medical-blue" />
              Clinical Assessment
            </CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="bg-blue-50 border border-blue-100 rounded-lg p-4">
                <div className="text-xs font-semibold text-blue-600 uppercase tracking-wide mb-2">Wound Type</div>
                <div className="text-gray-900 font-medium">{(assessmentData?.classification as any)?.woundType || 'Not specified'}</div>
              </div>
              <div className="bg-green-50 border border-green-100 rounded-lg p-4">
                <div className="text-xs font-semibold text-green-600 uppercase tracking-wide mb-2">Wound Stage</div>
                <div className="text-gray-900 font-medium">{(assessmentData?.classification as any)?.stage || 'Not applicable'}</div>
              </div>
              <div className="bg-purple-50 border border-purple-100 rounded-lg p-4">
                <div className="text-xs font-semibold text-purple-600 uppercase tracking-wide mb-2">Size Assessment</div>
                <div className="text-gray-900 font-medium">{(assessmentData?.classification as any)?.size || 'Not specified'}</div>
                {/* Show precise measurements if available from detection */}
                {(assessmentData?.classification as any)?.detection?.measurements && (
                  <div className="mt-2 pt-2 border-t border-purple-200">
                    <div className="text-xs text-purple-700 space-y-1">
                      {(assessmentData?.classification as any)?.detection?.measurements?.length_mm && (
                        <div>Length: {Math.round(Number((assessmentData?.classification as any)?.detection?.measurements?.length_mm))} mm</div>
                      )}
                      {(assessmentData?.classification as any)?.detection?.measurements?.width_mm && (
                        <div>Width: {Math.round(Number((assessmentData?.classification as any)?.detection?.measurements?.width_mm))} mm</div>
                      )}
                      {(assessmentData?.classification as any)?.detection?.measurements?.area_mm2 && (
                        <div>Area: {Math.round(Number((assessmentData?.classification as any)?.detection?.measurements?.area_mm2))} mm²</div>
                      )}
                      {(assessmentData?.classification as any)?.detection?.measurements?.perimeter_mm && (
                        <div>Perimeter: {Math.round(Number((assessmentData?.classification as any)?.detection?.measurements?.perimeter_mm))} mm</div>
                      )}
                    </div>
                  </div>
                )}
                {/* Show precise measurements if available from preciseMeasurements */}
                {(assessmentData?.classification as any)?.preciseMeasurements && (
                  <div className="mt-2 pt-2 border-t border-purple-200">
                    <div className="text-xs text-purple-700 space-y-1">
                      {(assessmentData?.classification as any)?.preciseMeasurements?.length_mm && (
                        <div>Length: {Math.round(Number((assessmentData?.classification as any)?.preciseMeasurements?.length_mm))} mm</div>
                      )}
                      {(assessmentData?.classification as any)?.preciseMeasurements?.width_mm && (
                        <div>Width: {Math.round(Number((assessmentData?.classification as any)?.preciseMeasurements?.width_mm))} mm</div>
                      )}
                      {(assessmentData?.classification as any)?.preciseMeasurements?.area_mm2 && (
                        <div>Area: {Math.round(Number((assessmentData?.classification as any)?.preciseMeasurements?.area_mm2))} mm²</div>
                      )}
                      {(assessmentData?.classification as any)?.preciseMeasurements?.perimeter_mm && (
                        <div>Perimeter: {Math.round(Number((assessmentData?.classification as any)?.preciseMeasurements?.perimeter_mm))} mm</div>
                      )}
                    </div>
                  </div>
                )}
              </div>
              <div className="bg-orange-50 border border-orange-100 rounded-lg p-4">
                <div className="text-xs font-semibold text-orange-600 uppercase tracking-wide mb-2">Anatomical Location</div>
                <div className="text-gray-900 font-medium">{(assessmentData?.classification as any)?.location || 'Not specified'}</div>
              </div>
              <div className="bg-yellow-50 border border-yellow-100 rounded-lg p-4">
                <div className="text-xs font-semibold text-yellow-600 uppercase tracking-wide mb-2">Exudate Level</div>
                <div className="text-gray-900 font-medium">{(assessmentData?.classification as any)?.exudate || 'Not assessed'}</div>
              </div>
              <div className="bg-pink-50 border border-pink-100 rounded-lg p-4">
                <div className="text-xs font-semibold text-pink-600 uppercase tracking-wide mb-2">Tissue Type</div>
                <div className="text-gray-900 font-medium">{(assessmentData?.classification as any)?.tissueType || 'Not specified'}</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Patient Context */}
        {assessmentData?.contextData && (() => {
          // Parse contextData if it's a string
          let contextData;
          try {
            contextData = typeof assessmentData.contextData === 'string' 
              ? JSON.parse(assessmentData.contextData)
              : assessmentData.contextData;
          } catch (e) {
            contextData = assessmentData.contextData;
          }
          
          // Check if contextData has meaningful content
          const hasContent = contextData && Object.values(contextData).some(value => 
            value && value !== '' && value !== 'Not provided' && value !== 'not provided'
          );
          
          if (!hasContent) return null;
          
          return (
            <Card className="mb-8 shadow-md">
              <CardHeader className="bg-slate-50 border-b">
                <CardTitle className="flex items-center text-lg">
                  <User className="h-5 w-5 mr-3 text-medical-blue" />
                  Patient Context & History
                </CardTitle>
              </CardHeader>
              <CardContent className="p-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(contextData as Record<string, any>).map(([key, value]) => {
                    if (!value || value === 'Not provided' || value === 'not provided' || value === '') return null;
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
                      stressLevel: 'Stress Level',
                      medicalHistory: 'Medical History',
                      woundChanges: 'Wound Changes',
                      currentCare: 'Current Care',
                      woundPain: 'Pain Level',
                      supportAtHome: 'Support at Home'
                    };
                    return (
                      <div key={key} className="bg-gray-50 border border-gray-200 rounded-lg p-4 border-l-4 border-l-medical-blue">
                        <div className="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-2">{labels[key] || key}</div>
                        <div className="text-gray-900 text-sm leading-relaxed">{value}</div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          );
        })()}

        {/* Evidence-Based Care Plan */}
        <Card className="mb-8 shadow-lg border-0 bg-white">
          <CardHeader className="bg-gradient-to-r from-medical-blue to-blue-600 text-white rounded-t-lg">
            <CardTitle className="text-xl font-semibold flex items-center">
              <FileText className="h-6 w-6 mr-3" />
              Evidence-Based Care Plan
              <Badge variant="secondary" className="ml-auto bg-white bg-opacity-20 text-white border-0">
                {assessmentData?.model?.toUpperCase() || 'AI Generated'}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-8">
            <div className="space-y-6">
              {formatCarePlan(assessmentData?.carePlan || '')}
            </div>
          </CardContent>
        </Card>

        {/* Medical Disclaimer */}
        <Card className="mb-8 shadow-md border-2 border-amber-200 bg-gradient-to-r from-amber-50 to-yellow-50">
          <CardContent className="p-6">
            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0">
                <AlertTriangle className="h-8 w-8 text-amber-600" />
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-amber-800 mb-3">Important Medical Disclaimer</h3>
                <p className="text-amber-700 leading-relaxed text-sm">
                  This is an AI-generated care plan based on image analysis and provided context. This assessment is for educational and informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional before implementing any wound care recommendations.
                </p>
                <div className="mt-4 p-3 bg-amber-100 rounded-lg border border-amber-200">
                  <p className="text-xs text-amber-700 font-medium">
                    ⚠️ Seek immediate medical attention if you notice signs of infection, worsening condition, or any concerning changes in the wound.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Professional Footer */}
        <div className="footer">
          <p className="footer-title">Generated by Wound Nurses AI Assessment System</p>
          <p className="footer-details">Report Date: {new Date().toLocaleDateString()} | Case ID: {caseId}</p>
          <p className="footer-confidential">This report contains confidential medical information</p>
          
          {/* Detection Analysis - Small font at bottom */}
          {renderDetectionAnalysis(assessmentData?.carePlan || '')}
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

      {/* Edit Case Name Dialog */}
      <Dialog open={isEditingCaseName} onOpenChange={setIsEditingCaseName}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Edit Case Name</DialogTitle>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <label htmlFor="caseName" className="text-sm font-medium">
                Case Name
              </label>
              <Input
                id="caseName"
                value={editedCaseName}
                onChange={(e) => setEditedCaseName(e.target.value)}
                placeholder="Enter a name for this case"
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    handleSaveCaseName();
                  }
                  if (e.key === 'Escape') {
                    handleCancelEditCaseName();
                  }
                }}
              />
              <p className="text-xs text-gray-500">
                Case ID: {caseId}
              </p>
            </div>
          </div>
          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={handleCancelEditCaseName}
              disabled={updateCaseNameMutation.isPending}
            >
              <X className="mr-2 h-4 w-4" />
              Cancel
            </Button>
            <Button 
              onClick={handleSaveCaseName}
              disabled={updateCaseNameMutation.isPending || !editedCaseName.trim()}
            >
              <Save className="mr-2 h-4 w-4" />
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}