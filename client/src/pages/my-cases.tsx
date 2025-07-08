import { useQuery, useMutation } from "@tanstack/react-query";
import { useAuth } from "@/hooks/useAuth";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Calendar, FileText, User, MapPin, Stethoscope, Circle, Plus, Settings, MoreVertical, Trash2, Download, ExternalLink, RefreshCw, Shield, Edit3, Save, X } from "lucide-react";
import { Link, useLocation } from "wouter";
import React, { useEffect } from "react";
import ImageDetectionStatus from "@/components/ImageDetectionStatus";
import { useToast } from "@/hooks/use-toast";
import { isUnauthorizedError } from "@/lib/authUtils";
import { apiRequest, queryClient } from "@/lib/queryClient";

export default function MyCases() {
  const { toast } = useToast();
  const { user, isAuthenticated, isLoading: authLoading } = useAuth();
  const [, setLocation] = useLocation();
  const [isEditingCaseName, setIsEditingCaseName] = React.useState(false);
  const [editedCaseName, setEditedCaseName] = React.useState("");
  const [editingCaseId, setEditingCaseId] = React.useState("");

  // Refresh data when component mounts or when user navigates back
  useEffect(() => {
    refetch();
  }, []);

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      toast({
        title: "Unauthorized",
        description: "Please log in to view your cases.",
        variant: "destructive",
      });
      setTimeout(() => {
        setLocation("/login");
      }, 500);
      return;
    }
  }, [isAuthenticated, authLoading, toast, setLocation]);

  const { data: cases, isLoading, error, refetch } = useQuery({
    queryKey: ["/api/my-cases"],
    retry: false,
    refetchOnWindowFocus: true,
    refetchOnMount: true,
    staleTime: 0, // Always consider data stale to ensure fresh data
  });

  // Group assessments by case ID and sort by version
  const groupedCases = React.useMemo(() => {
    if (!cases || (cases as any[]).length === 0) return {};
    
    const grouped = (cases as any[]).reduce((acc: any, assessment: any) => {
      if (!acc[assessment.caseId]) {
        acc[assessment.caseId] = [];
      }
      acc[assessment.caseId].push(assessment);
      return acc;
    }, {});
    
    // Sort assessments within each case by version number (newest first)
    Object.keys(grouped).forEach(caseId => {
      grouped[caseId].sort((a: any, b: any) => b.versionNumber - a.versionNumber);
    });
    
    return grouped;
  }, [cases]);

  // Delete mutation
  const deleteAssessmentMutation = useMutation({
    mutationFn: async (caseId: string) => {
      return await apiRequest("DELETE", `/api/assessment/${caseId}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/my-cases"] });
      toast({
        title: "Case Deleted",
        description: "The wound assessment case has been deleted successfully.",
      });
    },
    onError: (error) => {
      if (isUnauthorizedError(error as Error)) {
        toast({
          title: "Unauthorized",
          description: "You are logged out. Logging in again...",
          variant: "destructive",
        });
        setTimeout(() => {
          window.location.href = "/api/login";
        }, 500);
        return;
      }
      toast({
        title: "Error",
        description: "Failed to delete the case. Please try again.",
        variant: "destructive",
      });
    },
  });

  // Update case name mutation
  const updateCaseNameMutation = useMutation({
    mutationFn: async ({ caseId, caseName }: { caseId: string; caseName: string }) => {
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
      setEditedCaseName("");
      setEditingCaseId("");
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

  const handleDeleteCase = (caseId: string) => {
    if (window.confirm("Are you sure you want to delete this case? This action cannot be undone.")) {
      deleteAssessmentMutation.mutate(caseId);
    }
  };

  // Functions to handle case name editing
  const handleEditCaseName = (caseId: string, currentName: string) => {
    setEditingCaseId(caseId);
    setEditedCaseName(currentName || "");
    setIsEditingCaseName(true);
  };

  const handleSaveCaseName = () => {
    if (editedCaseName.trim() && editingCaseId) {
      updateCaseNameMutation.mutate({
        caseId: editingCaseId,
        caseName: editedCaseName.trim()
      });
    }
  };

  const handleCancelEditCaseName = () => {
    setIsEditingCaseName(false);
    setEditedCaseName("");
    setEditingCaseId("");
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-medical-blue to-medical-teal flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-white"></div>
      </div>
    );
  }

  if (error && isUnauthorizedError(error as Error)) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-medical-blue to-medical-teal flex items-center justify-center">
        <div className="text-center text-white">
          <h2 className="text-2xl font-bold mb-4">Authentication Required</h2>
          <p className="mb-4">Please log in to view your cases.</p>
          <Button onClick={() => setLocation("/login")} className="bg-white text-medical-blue hover:bg-gray-100">
            Log In
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-bg-light to-gray-50">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <Link href="/">
                <div className="flex items-center cursor-pointer">
                  <Stethoscope className="text-medical-blue text-2xl mr-3" />
                  <span className="text-xl font-bold text-gray-900">Wound Nurses</span>
                </div>
              </Link>
            </div>
            <div className="flex items-center gap-3">
              <div className="text-sm text-gray-500">
                <span className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs font-medium">
                  <Circle className="inline w-2 h-2 mr-1 fill-current" />
                  System Online
                </span>
              </div>
              <ImageDetectionStatus />
              <Button 
                variant="outline"
                onClick={() => setLocation("/assessment")}
                className="border-medical-blue text-medical-blue hover:bg-medical-blue hover:text-white"
              >
                New Assessment
              </Button>
              <Button 
                variant="outline"
                onClick={() => refetch()}
                className="border-gray-300 text-gray-700 hover:bg-gray-50"
                title="Refresh Cases"
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh
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
              {(user as any)?.role === 'admin' && (
                <Button 
                  variant="ghost"
                  size="sm"
                  onClick={() => setLocation("/admin")}
                  className="p-2 text-red-600 hover:text-red-700 hover:bg-red-50"
                  title="Admin Dashboard"
                >
                  <Shield className="h-4 w-4" />
                </Button>
              )}
              <Button 
                variant="ghost"
                onClick={() => {
                  localStorage.removeItem('auth_token');
                  setLocation('/');
                }}
              >
                Log Out
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        {/* Page Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">My Wound Care Cases</h1>
          <p className="text-gray-600 text-lg">Manage and review your Care Plans</p>
        </div>

        {/* Cases Grid */}
        <div className="max-w-6xl mx-auto">
          {isLoading ? (
            <div className="text-center text-white">
              <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-white mx-auto mb-4"></div>
              Loading your cases...
            </div>
          ) : !cases || (cases as any[]).length === 0 ? (
            <Card className="bg-white/95 backdrop-blur-sm">
              <CardContent className="text-center py-12">
                <FileText className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-gray-900 mb-2">No Cases Yet</h3>
                <p className="text-gray-600 mb-6">
                  You haven't created any wound assessments yet. Start by uploading your first wound image.
                </p>
                <Link href="/assessment">
                  <Button className="bg-medical-blue hover:bg-medical-blue/90">
                    Create First Assessment
                  </Button>
                </Link>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Object.entries(groupedCases).map(([caseId, assessments]: [string, any]) => {
                const latestAssessment = assessments[0]; // Most recent version
                const originalAssessment = assessments[assessments.length - 1]; // Original version
                
                return (
                  <Card key={caseId} className="bg-white/95 backdrop-blur-sm hover:bg-white transition-all duration-200">
                    <CardHeader>
                      <CardTitle className="flex items-center justify-between">
                        <div className="flex flex-col">
                          <span className="text-sm font-medium">
                            {latestAssessment.caseName || `Case ${caseId}`}
                          </span>
                          {latestAssessment.caseName && (
                            <span className="text-xs text-gray-500 font-normal">
                              Case ID: {caseId}
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant={originalAssessment.audience === 'medical' ? 'default' : 'secondary'}>
                            {originalAssessment.audience}
                          </Badge>
                          {assessments.length > 1 && (
                            <Badge variant="outline" className="text-xs">
                              v{latestAssessment.versionNumber}
                            </Badge>
                          )}
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                                <MoreVertical className="h-4 w-4" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              <DropdownMenuItem asChild>
                                <Link href={`/care-plan/${caseId}`} className="flex items-center cursor-pointer">
                                  <ExternalLink className="h-4 w-4 mr-2" />
                                  View Care Plan
                                </Link>
                              </DropdownMenuItem>
                              <DropdownMenuItem
                                onClick={() => handleEditCaseName(caseId, latestAssessment.caseName)}
                                className="cursor-pointer"
                              >
                                <Edit3 className="h-4 w-4 mr-2" />
                                Edit Case Name
                              </DropdownMenuItem>
                              <DropdownMenuItem asChild>
                                <Link href={`/follow-up/${caseId}`} className="flex items-center cursor-pointer">
                                  <Plus className="h-4 w-4 mr-2" />
                                  Add Follow-up
                                </Link>
                              </DropdownMenuItem>
                              <DropdownMenuSeparator />
                              <DropdownMenuItem
                                onClick={() => handleDeleteCase(caseId)}
                                className="text-red-600 cursor-pointer"
                              >
                                <Trash2 className="h-4 w-4 mr-2" />
                                Delete Case
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </div>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      {/* Latest Wound Image */}
                      {latestAssessment.imageData && (
                        <div className="mb-4">
                          <img
                            src={`data:${latestAssessment.imageMimeType};base64,${latestAssessment.imageData}`}
                            alt="Latest wound assessment"
                            className="w-full h-32 object-cover rounded-lg border border-gray-200"
                          />
                        </div>
                      )}

                      {/* Assessment History */}
                      <div className="space-y-3 mb-4">
                        {assessments.map((assessment: any, index: number) => (
                          <Link key={assessment.id} href={`/care-plan/${caseId}?version=${assessment.versionNumber}`}>
                            <div className={`text-xs p-2 rounded cursor-pointer transition-colors hover:bg-opacity-80 ${index === 0 ? 'bg-blue-50 border border-blue-200 hover:bg-blue-100' : 'bg-gray-50 hover:bg-gray-100'}`}>
                              <div className="flex items-center justify-between mb-1">
                                <span className="font-medium">
                                  {assessment.isFollowUp ? `Follow-up / ${new Date(assessment.createdAt).toLocaleDateString()}` : 'Original Assessment'}
                                </span>
                                <span className="text-gray-500 text-xs">
                                  v{assessment.versionNumber}
                                </span>
                              </div>
                              {assessment.classification?.woundType && (
                                <div className="text-gray-600">
                                  {assessment.classification.woundType}
                                </div>
                              )}
                              {assessment.progressNotes && (
                                <div className="text-gray-600 mt-1 italic">
                                  "{assessment.progressNotes.substring(0, 50)}..."
                                </div>
                              )}
                            </div>
                          </Link>
                        ))}
                      </div>

                      {/* Action Buttons */}
                      <div className="space-y-2">
                        <Link href={`/care-plan/${caseId}`}>
                          <Button size="sm" className="w-full bg-medical-blue hover:bg-medical-blue/90">
                            <FileText className="h-4 w-4 mr-2" />
                            View Latest Care Plan
                          </Button>
                        </Link>
                        <Link href={`/follow-up/${caseId}`}>
                          <Button size="sm" variant="outline" className="w-full border-green-200 text-green-600 hover:bg-green-50 hover:border-green-300">
                            <Plus className="h-4 w-4 mr-2" />
                            Add Follow-up Assessment
                          </Button>
                        </Link>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          )}
        </div>
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
                Case ID: {editingCaseId}
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