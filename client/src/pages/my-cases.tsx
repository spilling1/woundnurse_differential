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
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Calendar, FileText, User, MapPin, Stethoscope, Circle, Plus, Settings, MoreVertical, Trash2, Download, ExternalLink, RefreshCw, Shield, Edit3, Save, X, Search, ChevronDown, ChevronUp, SortAsc, SortDesc, BarChart3 } from "lucide-react";
import { Link, useLocation } from "wouter";
import React, { useEffect, useState, useMemo } from "react";
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
  
  // Search and sort state
  const [searchTerm, setSearchTerm] = useState("");
  const [sortBy, setSortBy] = useState<"name" | "date" | "woundType">("date");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [isSearchOpen, setIsSearchOpen] = useState(false);

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

  // Filter and sort case IDs based on search and sort criteria
  const filteredAndSortedCaseIds = useMemo(() => {
    let caseIds = Object.keys(groupedCases);

    // Apply search filter
    if (searchTerm.trim()) {
      const term = searchTerm.toLowerCase();
      caseIds = caseIds.filter(caseId => {
        const latestAssessment = groupedCases[caseId][0]; // First item is the latest
        const caseName = latestAssessment.caseName?.toLowerCase() || '';
        const caseIdLower = caseId.toLowerCase();
        const audience = latestAssessment.audience?.toLowerCase() || '';
        
        // Search in classification object if it exists
        let classificationText = '';
        if (latestAssessment.classification) {
          const classification = typeof latestAssessment.classification === 'string' 
            ? latestAssessment.classification 
            : JSON.stringify(latestAssessment.classification);
          classificationText = classification.toLowerCase();
        }
        
        return caseName.includes(term) || 
               caseIdLower.includes(term) || 
               audience.includes(term) ||
               classificationText.includes(term);
      });
    }

    // Apply sorting
    caseIds.sort((a, b) => {
      const aLatest = groupedCases[a][0];
      const bLatest = groupedCases[b][0];
      
      let aValue: string | Date;
      let bValue: string | Date;

      switch (sortBy) {
        case 'name':
          aValue = aLatest.caseName || a;
          bValue = bLatest.caseName || b;
          break;
        case 'date':
          aValue = new Date(aLatest.createdAt);
          bValue = new Date(bLatest.createdAt);
          break;
        case 'woundType':
          // Extract wound type from classification object
          aValue = '';
          bValue = '';
          if (aLatest.classification) {
            const aClassification = typeof aLatest.classification === 'string' 
              ? aLatest.classification 
              : JSON.stringify(aLatest.classification);
            aValue = aClassification;
          }
          if (bLatest.classification) {
            const bClassification = typeof bLatest.classification === 'string' 
              ? bLatest.classification 
              : JSON.stringify(bLatest.classification);
            bValue = bClassification;
          }
          break;
        default:
          aValue = new Date(aLatest.createdAt);
          bValue = new Date(bLatest.createdAt);
      }

      if (sortBy === 'date') {
        const result = (aValue as Date).getTime() - (bValue as Date).getTime();
        return sortOrder === 'asc' ? result : -result;
      } else {
        const result = (aValue as string).localeCompare(bValue as string);
        return sortOrder === 'asc' ? result : -result;
      }
    });

    return caseIds;
  }, [groupedCases, searchTerm, sortBy, sortOrder]);

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

        {/* Search and Sort Controls - Only show if more than 9 cases */}
        {Object.keys(groupedCases).length > 9 && (
          <div className="max-w-6xl mx-auto mb-6">
            <Collapsible open={isSearchOpen} onOpenChange={setIsSearchOpen}>
              <CollapsibleTrigger asChild>
                <Button 
                  variant="outline" 
                  className="w-full bg-white/95 backdrop-blur-sm hover:bg-white border-gray-300 mb-4"
                >
                  <Search className="h-4 w-4 mr-2" />
                  Search & Sort Cases ({Object.keys(groupedCases).length} total)
                  {isSearchOpen ? <ChevronUp className="h-4 w-4 ml-2" /> : <ChevronDown className="h-4 w-4 ml-2" />}
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Search Card */}
                  <Card className="bg-gray-100 border-gray-200 shadow-sm md:col-span-2">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium text-gray-700">Search Cases</CardTitle>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <div className="relative">
                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                        <Input
                          placeholder="Search cases by name, ID, wound type, or audience..."
                          value={searchTerm}
                          onChange={(e) => setSearchTerm(e.target.value)}
                          className="pl-10 bg-white"
                        />
                      </div>
                      
                      {/* Search Results Summary */}
                      {searchTerm && (
                        <div className="mt-3 pt-3 border-t border-gray-300">
                          <p className="text-sm text-gray-600">
                            Showing {filteredAndSortedCaseIds.length} of {Object.keys(groupedCases).length} cases
                          </p>
                        </div>
                      )}
                    </CardContent>
                  </Card>

                  {/* Sort Card */}
                  <Card className="bg-gray-100 border-gray-200 shadow-sm md:col-span-1">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium text-gray-700">Sort Cases</CardTitle>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <div className="flex flex-wrap gap-2 items-center">
                        <span className="text-sm text-gray-600 whitespace-nowrap">Sort by:</span>
                        <Button
                          variant={sortBy === 'name' ? 'default' : 'outline'}
                          size="sm"
                          onClick={() => setSortBy('name')}
                          className="text-xs"
                        >
                          Name
                        </Button>
                        <Button
                          variant={sortBy === 'date' ? 'default' : 'outline'}
                          size="sm"
                          onClick={() => setSortBy('date')}
                          className="text-xs"
                        >
                          Date
                        </Button>
                        <Button
                          variant={sortBy === 'woundType' ? 'default' : 'outline'}
                          size="sm"
                          onClick={() => setSortBy('woundType')}
                          className="text-xs"
                        >
                          Type
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                          className="px-2"
                          title={sortOrder === 'asc' ? 'Sort ascending' : 'Sort descending'}
                        >
                          {sortOrder === 'asc' ? <SortAsc className="h-4 w-4" /> : <SortDesc className="h-4 w-4" />}
                        </Button>
                      </div>
                      
                      {/* Sort Status */}
                      {sortBy !== 'date' && (
                        <div className="mt-3 pt-3 border-t border-gray-300">
                          <p className="text-sm text-gray-600">
                            Sorted by {sortBy === 'name' ? 'case name' : sortBy === 'date' ? 'date' : 'wound type'} ({sortOrder === 'asc' ? 'ascending' : 'descending'})
                          </p>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </div>
              </CollapsibleContent>
            </Collapsible>
          </div>
        )}

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
          ) : filteredAndSortedCaseIds.length === 0 && searchTerm ? (
            <Card className="bg-white/95 backdrop-blur-sm">
              <CardContent className="text-center py-12">
                <Search className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-gray-900 mb-2">No Cases Found</h3>
                <p className="text-gray-600 mb-4">
                  No cases match your search for "{searchTerm}". Try adjusting your search terms.
                </p>
                <Button 
                  variant="outline" 
                  onClick={() => setSearchTerm("")}
                  className="mr-2"
                >
                  Clear Search
                </Button>
                <Button 
                  variant="outline" 
                  onClick={() => setIsSearchOpen(false)}
                >
                  Close Search
                </Button>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredAndSortedCaseIds.map((caseId) => {
                const assessments = groupedCases[caseId];
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
                                <Link href={`/case-analysis/${caseId}`} className="flex items-center cursor-pointer">
                                  <BarChart3 className="h-4 w-4 mr-2" />
                                  Detailed Analysis
                                </Link>
                              </DropdownMenuItem>
                              <DropdownMenuItem
                                onClick={() => handleEditCaseName(caseId, latestAssessment.caseName)}
                                className="cursor-pointer"
                              >
                                <Edit3 className="h-4 w-4 mr-2" />
                                Edit Case Name
                              </DropdownMenuItem>

                              {user?.role === 'admin' && (
                                <DropdownMenuItem asChild>
                                  <Link href={`/admin/analysis/${caseId}`} className="flex items-center cursor-pointer">
                                    <BarChart3 className="h-4 w-4 mr-2" />
                                    Admin Analysis
                                  </Link>
                                </DropdownMenuItem>
                              )}
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