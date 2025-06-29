import { useQuery, useMutation } from "@tanstack/react-query";
import { useAuth } from "@/hooks/useAuth";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Calendar, FileText, User, MapPin, Stethoscope, Circle } from "lucide-react";
import { Link, useLocation } from "wouter";
import { useEffect } from "react";
import { useToast } from "@/hooks/use-toast";
import { isUnauthorizedError } from "@/lib/authUtils";
import { apiRequest, queryClient } from "@/lib/queryClient";

export default function MyCases() {
  const { toast } = useToast();
  const { isAuthenticated, isLoading: authLoading } = useAuth();
  const [, setLocation] = useLocation();

  // Temporarily disable auth check to stop infinite loop
  // TODO: Re-enable after fixing useAuth hook
  /*
  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      toast({
        title: "Unauthorized",
        description: "Please log in to view your cases.",
        variant: "destructive",
      });
      setTimeout(() => {
        window.location.href = "/api/login";
      }, 500);
      return;
    }
  }, [isAuthenticated, authLoading, toast]);
  */

  const { data: cases, isLoading, error } = useQuery({
    queryKey: ["/api/my-cases"],
    enabled: isAuthenticated,
  });

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

  const handleDeleteCase = (caseId: string) => {
    if (window.confirm("Are you sure you want to delete this case? This action cannot be undone.")) {
      deleteAssessmentMutation.mutate(caseId);
    }
  };

  if (authLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-medical-blue to-medical-teal flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-white"></div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return null; // Redirect happening
  }

  if (error && isUnauthorizedError(error as Error)) {
    return null; // Redirect happening
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
              <Button 
                variant="outline"
                onClick={() => setLocation("/assessment")}
                className="border-medical-blue text-medical-blue hover:bg-medical-blue hover:text-white"
              >
                New Assessment
              </Button>
              <Button 
                variant="ghost"
                onClick={() => window.location.href = "/api/logout"}
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
          ) : !cases || cases.length === 0 ? (
            <Card className="bg-white/95 backdrop-blur-sm">
              <CardContent className="text-center py-12">
                <FileText className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-gray-900 mb-2">No Cases Yet</h3>
                <p className="text-gray-600 mb-6">
                  You haven't created any wound assessments yet. Start by uploading your first wound image.
                </p>
                <Link href="/">
                  <Button className="bg-medical-blue hover:bg-medical-blue/90">
                    Create First Assessment
                  </Button>
                </Link>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {cases.map((assessment: any) => (
                <Card key={assessment.caseId} className="bg-white/95 backdrop-blur-sm hover:bg-white transition-all duration-200">
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      <span className="text-sm font-medium">Case {assessment.caseId}</span>
                      <Badge variant={assessment.audience === 'medical' ? 'default' : 'secondary'}>
                        {assessment.audience}
                      </Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {/* Wound Image */}
                    {assessment.imageData && (
                      <div className="mb-4">
                        <img
                          src={`data:${assessment.imageMimeType};base64,${assessment.imageData}`}
                          alt="Wound assessment"
                          className="w-full h-32 object-cover rounded-lg border border-gray-200"
                        />
                      </div>
                    )}

                    {/* Assessment Details */}
                    <div className="space-y-2 text-sm">
                      <div className="flex items-center text-gray-600">
                        <Calendar className="h-4 w-4 mr-2" />
                        {new Date(assessment.createdAt).toLocaleDateString()}
                      </div>
                      
                      {assessment.classification?.woundType && (
                        <div className="flex items-center text-gray-600">
                          <MapPin className="h-4 w-4 mr-2" />
                          {assessment.classification.woundType}
                        </div>
                      )}
                    </div>

                    {/* Action Buttons */}
                    <div className="mt-4 space-y-2">
                      <Link href={`/care-plan/${assessment.caseId}`}>
                        <Button size="sm" className="w-full bg-medical-blue hover:bg-medical-blue/90">
                          <FileText className="h-4 w-4 mr-2" />
                          View Care Plan
                        </Button>
                      </Link>
                      <Button 
                        size="sm" 
                        variant="outline"
                        className="w-full text-red-600 border-red-200 hover:bg-red-50 hover:border-red-300"
                        onClick={() => handleDeleteCase(assessment.caseId)}
                      >
                        Delete Case
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}