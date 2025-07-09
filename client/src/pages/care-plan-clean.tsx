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
      // Create a clone of the element to avoid modifying the original
      const elementToCapture = printRef.current.cloneNode(true) as HTMLElement;
      
      // Apply PDF-specific styles
      elementToCapture.style.width = '8.5in';
      elementToCapture.style.padding = '0.5in';
      elementToCapture.style.backgroundColor = 'white';
      elementToCapture.style.fontFamily = 'Arial, sans-serif';
      elementToCapture.style.fontSize = '12px';
      elementToCapture.style.color = '#000000';
      
      // Hide the element off-screen
      elementToCapture.style.position = 'absolute';
      elementToCapture.style.left = '-9999px';
      elementToCapture.style.top = '-9999px';
      
      // Add to document temporarily
      document.body.appendChild(elementToCapture);
      
      // Generate canvas from the element
      const canvas = await html2canvas(elementToCapture, {
        width: 816, // 8.5in * 96 DPI
        height: 1056, // 11in * 96 DPI
        scale: 2,
        useCORS: true,
        allowTaint: true,
        backgroundColor: '#ffffff'
      });
      
      // Remove the temporary element
      document.body.removeChild(elementToCapture);
      
      // Create PDF
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'in',
        format: 'letter'
      });
      
      // Add title page
      pdf.setFontSize(20);
      pdf.text('Wound Care Assessment Report', 4.25, 1.5, { align: 'center' });
      
      pdf.setFontSize(12);
      pdf.text(`Case ID: ${assessmentData.caseId}`, 4.25, 2.2, { align: 'center' });
      pdf.text(`Generated: ${new Date().toLocaleDateString()}`, 4.25, 2.5, { align: 'center' });
      pdf.text(`Assessment Date: ${new Date(assessmentData.createdAt).toLocaleDateString()}`, 4.25, 2.8, { align: 'center' });
      
      // Add the image content
      const imgData = canvas.toDataURL('image/png');
      const imgWidth = 8;
      const imgHeight = (canvas.height * imgWidth) / canvas.width;
      
      pdf.addPage();
      pdf.addImage(imgData, 'PNG', 0.5, 0.5, imgWidth, imgHeight);
      
      // Add footer
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
