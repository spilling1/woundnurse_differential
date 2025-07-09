import { useLocation, useParams, useSearch } from "wouter";
import { ArrowLeft, ClipboardList, AlertTriangle, ThumbsUp, ThumbsDown, Download, Printer, UserCheck, Calendar, MapPin, User, FileText, Plus, LogOut, Settings, RefreshCw, MoreVertical, Edit3, Save, X } from "lucide-react";
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
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

  const handleDownloadPDF = async () => {
    if (!printRef.current || !assessmentData) return;
    
    try {
      // Create PDF
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'in',
        format: 'letter'
      });
      
      // Add title page with wound image
      pdf.setFontSize(20);
      pdf.text('Wound Care Assessment Report', 4.25, 1.5, { align: 'center' });
      
      pdf.setFontSize(12);
      pdf.text(`Case ID: ${assessmentData.caseId}`, 4.25, 2.2, { align: 'center' });
      pdf.text(`Generated: ${new Date().toLocaleDateString()}`, 4.25, 2.5, { align: 'center' });
      pdf.text(`Assessment Date: ${new Date(assessmentData.createdAt).toLocaleDateString()}`, 4.25, 2.8, { align: 'center' });
      
      // Add wound image to title page if available with compression
      if (assessmentData.imageData) {
        const imgData = assessmentData.imageData.startsWith('data:') 
          ? assessmentData.imageData 
          : `data:image/jpeg;base64,${assessmentData.imageData}`;
        
        // Add image below the title information with compression
        try {
          // Create a compressed version of the image
          const compressedImageData = await new Promise<string>((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
              const canvas = document.createElement('canvas');
              const ctx = canvas.getContext('2d');
              
              // Set canvas size for compression (reduce from original)
              const maxWidth = 800;
              const maxHeight = 600;
              let { width, height } = img;
              
              if (width > maxWidth || height > maxHeight) {
                if (width > height) {
                  height = (height * maxWidth) / width;
                  width = maxWidth;
                } else {
                  width = (width * maxHeight) / height;
                  height = maxHeight;
                }
              }
              
              canvas.width = width;
              canvas.height = height;
              
              ctx.drawImage(img, 0, 0, width, height);
              
              // Convert to JPEG with compression
              const compressed = canvas.toDataURL('image/jpeg', 0.7); // 70% quality
              resolve(compressed);
            };
            img.onerror = reject;
            img.src = imgData;
          });
          
          const imgWidth = 5; // 5-inch wide image
          const imgHeight = 3.5; // 3.5-inch tall image
          const x = (8.5 - imgWidth) / 2; // Center horizontally
          const y = 3.5; // Position below title info
          
          pdf.addImage(compressedImageData, 'JPEG', x, y, imgWidth, imgHeight);
        } catch (imgError) {
          console.warn('Could not add image to PDF:', imgError);
          // Fallback to original image if compression fails
          const imgWidth = 5;
          const imgHeight = 3.5;
          const x = (8.5 - imgWidth) / 2;
          const y = 3.5;
          
          pdf.addImage(imgData, 'JPEG', x, y, imgWidth, imgHeight);
        }
      }
      
      // Add care plan content using text rendering for smaller file size
      pdf.addPage();
      pdf.setFontSize(16);
      pdf.text('Care Plan', 0.5, 1);
      
      // Extract text content from the care plan
      const carePlanHtml = assessmentData.carePlan || '';
      
      // Parse the HTML and render as text
      const tempDiv = document.createElement('div');
      tempDiv.innerHTML = carePlanHtml;
      
      let currentY = 1.5;
      const pageHeight = 10.5; // Letter size height minus margins
      const lineHeight = 0.2;
      const marginLeft = 0.5;
      const marginRight = 0.5;
      const maxWidth = 7.5; // 8.5 - 1 inch margins
      
      // Function to add text with word wrapping
      const addTextWithWrapping = (text: string, fontSize: number, isBold: boolean = false, isRed: boolean = false) => {
        pdf.setFontSize(fontSize);
        
        if (isBold) {
          pdf.setFont('helvetica', 'bold');
        } else {
          pdf.setFont('helvetica', 'normal');
        }
        
        if (isRed) {
          pdf.setTextColor(255, 0, 0); // Red color for urgent messages
        } else {
          pdf.setTextColor(0, 0, 0); // Black color
        }
        
        const lines = pdf.splitTextToSize(text, maxWidth);
        
        for (const line of lines) {
          if (currentY > pageHeight - 1) {
            pdf.addPage();
            currentY = 1;
          }
          
          pdf.text(line, marginLeft, currentY);
          currentY += lineHeight;
        }
        
        pdf.setTextColor(0, 0, 0); // Reset to black
      };
      
      // Process the care plan content
      const processElement = (element: Element) => {
        if (element.tagName === 'H1') {
          addTextWithWrapping(element.textContent || '', 16, true);
          currentY += 0.1; // Extra spacing after headers
        } else if (element.tagName === 'H2') {
          addTextWithWrapping(element.textContent || '', 14, true);
          currentY += 0.1;
        } else if (element.tagName === 'H3') {
          addTextWithWrapping(element.textContent || '', 12, true);
          currentY += 0.1;
        } else if (element.tagName === 'P') {
          const isUrgent = element.innerHTML.includes('URGENT') || element.innerHTML.includes('MEDICAL EMERGENCY');
          addTextWithWrapping(element.textContent || '', 11, false, isUrgent);
          currentY += 0.1; // Extra spacing after paragraphs
        } else if (element.tagName === 'UL' || element.tagName === 'OL') {
          Array.from(element.children).forEach((li, index) => {
            const bullet = element.tagName === 'UL' ? '• ' : `${index + 1}. `;
            addTextWithWrapping(bullet + (li.textContent || ''), 11);
          });
          currentY += 0.1;
        } else if (element.tagName === 'DIV') {
          const isUrgent = element.innerHTML.includes('URGENT') || element.innerHTML.includes('MEDICAL EMERGENCY');
          if (element.textContent?.trim()) {
            addTextWithWrapping(element.textContent, 11, false, isUrgent);
          }
        }
        
        // Process child elements
        Array.from(element.children).forEach(child => {
          if (!['UL', 'OL', 'LI'].includes(child.tagName)) {
            processElement(child);
          }
        });
      };
      
      // Process all elements in the care plan
      Array.from(tempDiv.children).forEach(element => {
        processElement(element);
      });
      
      // If no structured content, fall back to plain text
      if (currentY <= 1.5) {
        const plainText = tempDiv.textContent || carePlanHtml;
        addTextWithWrapping(plainText, 11);
      }
      
      // Add footer to all pages
      const pageCount = pdf.internal.getNumberOfPages();
      for (let i = 1; i <= pageCount; i++) {
        pdf.setPage(i);
        pdf.setFontSize(9);
        pdf.text(`Page ${i} of ${pageCount}`, 4.25, 10.5, { align: 'center' });
        pdf.text('Generated by Wound Nurses AI Assessment Tool', 4.25, 10.8, { align: 'center' });
      }
      
      // Download the PDF
      const fileName = `wound-care-assessment-${assessmentData.caseId}-${new Date().toISOString().split('T')[0]}.pdf`;
      pdf.save(fileName);
      
      toast({
        title: "PDF Downloaded",
        description: "Your care plan has been downloaded as a PDF.",
      });
      
    } catch (error) {
      console.error('PDF generation error:', error);
      toast({
        title: "PDF Generation Failed",
        description: "There was an error generating the PDF. Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleEditCaseName = () => {
    setEditedCaseName(assessmentData?.case_name || assessmentData?.caseId || '');
    setIsEditingCaseName(true);
  };

  const handleSaveCaseName = () => {
    if (editedCaseName.trim()) {
      updateCaseNameMutation.mutate(editedCaseName.trim());
    }
  };

  const handleCancelEdit = () => {
    setIsEditingCaseName(false);
    setEditedCaseName('');
  };

  if (!isAuthenticated) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">Authentication Required</h1>
          <p className="text-gray-600 mb-6">Please log in to view your care plan.</p>
          <Button onClick={() => setLocation('/login')}>Log In</Button>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-medical-blue"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-red-600 mb-4">Error Loading Care Plan</h1>
          <p className="text-gray-600 mb-6">
            {error.message || 'There was an error loading your care plan.'}
          </p>
          <Button onClick={() => setLocation('/cases')}>Back to Cases</Button>
        </div>
      </div>
    );
  }

  if (!assessmentData) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">Care Plan Not Found</h1>
          <p className="text-gray-600 mb-6">The requested care plan could not be found.</p>
          <Button onClick={() => setLocation('/cases')}>Back to Cases</Button>
        </div>
      </div>
    );
  }

  // Safely parse JSON data - handle both string and object cases
  const classification = typeof assessmentData.classification === 'string' 
    ? JSON.parse(assessmentData.classification) 
    : assessmentData.classification;
  
  const contextData = typeof assessmentData.contextData === 'string'
    ? JSON.parse(assessmentData.contextData || '{}')
    : assessmentData.contextData || {};

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-6">
          <Button
            variant="outline"
            onClick={() => setLocation('/cases')}
            className="flex items-center gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Cases
          </Button>
          
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              onClick={handleDownloadPDF}
              className="flex items-center gap-2"
            >
              <Download className="h-4 w-4" />
              Download PDF
            </Button>
            
            {/* Settings Button */}
            <Button
              variant="outline"
              onClick={() => setLocation('/settings')}
              className="flex items-center gap-2"
            >
              <Settings className="h-4 w-4" />
            </Button>
            
            {/* Admin Navigation */}
            <AdminNavigation />
            
            {/* Logout Button */}
            <Button
              variant="outline"
              onClick={() => {
                localStorage.removeItem('authToken');
                setLocation('/login');
              }}
              className="flex items-center gap-2"
            >
              <LogOut className="h-4 w-4" />
              Logout
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div ref={printRef} className="space-y-6">
        {/* Case Header */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-2xl">
                  {assessmentData.case_name || `Case ${assessmentData.caseId}`}
                </CardTitle>
                <div className="flex items-center gap-4 mt-2 text-sm text-gray-600">
                  <div className="flex items-center gap-1">
                    <Calendar className="h-4 w-4" />
                    {new Date(assessmentData.createdAt).toLocaleDateString()}
                  </div>
                  <div className="flex items-center gap-1">
                    <FileText className="h-4 w-4" />
                    {assessmentData.caseId}
                  </div>
                  <div className="flex items-center gap-1">
                    <User className="h-4 w-4" />
                    {assessmentData.audience}
                  </div>
                </div>
              </div>
              
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" size="sm">
                    <MoreVertical className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent>
                  <DropdownMenuItem onClick={handleEditCaseName}>
                    <Edit3 className="h-4 w-4 mr-2" />
                    Edit Case Name
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => refetch()}>
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Refresh
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </CardHeader>
        </Card>

        {/* Wound Image */}
        {assessmentData.imageData && (
          <Card data-wound-image>
            <CardHeader>
              <CardTitle>Wound Image</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex justify-center">
                <img 
                  src={assessmentData.imageData.startsWith('data:') 
                    ? assessmentData.imageData 
                    : `data:image/jpeg;base64,${assessmentData.imageData}`}
                  alt="Wound assessment"
                  className="max-w-2xl max-h-96 object-contain rounded-lg shadow-lg cursor-pointer"
                  onClick={() => {
                    const modal = document.createElement('div');
                    modal.className = 'fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50';
                    modal.innerHTML = `
                      <div class="relative max-w-4xl max-h-full p-4">
                        <img src="${assessmentData.imageData.startsWith('data:') ? assessmentData.imageData : `data:image/jpeg;base64,${assessmentData.imageData}`}" 
                             alt="Wound assessment" 
                             class="max-w-full max-h-full object-contain rounded-lg" />
                        <button class="absolute top-2 right-2 text-white bg-black bg-opacity-50 rounded-full p-2 hover:bg-opacity-75">
                          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                          </svg>
                        </button>
                      </div>
                    `;
                    modal.onclick = (e) => {
                      if (e.target === modal || e.target.closest('button')) {
                        document.body.removeChild(modal);
                      }
                    };
                    document.body.appendChild(modal);
                  }}
                />
              </div>
            </CardContent>
          </Card>
        )}

        {/* Care Plan */}
        {assessmentData.carePlan && (
          <Card>
            <CardHeader>
              <CardTitle>Care Plan</CardTitle>
            </CardHeader>
            <CardContent>
              <div 
                className="prose prose-sm max-w-none" 
                dangerouslySetInnerHTML={{ 
                  __html: assessmentData.carePlan
                }}
              />
            </CardContent>
          </Card>
        )}
      </div>

      {/* Edit Case Name Dialog */}
      <Dialog open={isEditingCaseName} onOpenChange={setIsEditingCaseName}>
        <DialogContent>
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
                placeholder="Enter case name..."
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={handleCancelEdit}>
              <X className="h-4 w-4 mr-2" />
              Cancel
            </Button>
            <Button onClick={handleSaveCaseName} disabled={updateCaseNameMutation.isPending}>
              {updateCaseNameMutation.isPending ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="h-4 w-4 mr-2" />
                  Save
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
