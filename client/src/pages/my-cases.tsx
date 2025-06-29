import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@/hooks/useAuth";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Calendar, FileText, User, MapPin } from "lucide-react";
import { Link } from "wouter";
import { useEffect } from "react";
import { useToast } from "@/hooks/use-toast";
import { isUnauthorizedError } from "@/lib/authUtils";

export default function MyCases() {
  const { toast } = useToast();
  const { isAuthenticated, isLoading: authLoading } = useAuth();

  // Redirect to home if not authenticated
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

  const { data: cases, isLoading, error } = useQuery({
    queryKey: ["/api/my-cases"],
    enabled: isAuthenticated,
  });

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
    <div className="min-h-screen bg-gradient-to-br from-medical-blue to-medical-teal">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">My Wound Care Cases</h1>
          <p className="text-medical-light text-lg">Manage and review your Care Plans</p>
        </div>

        {/* Navigation */}
        <div className="mb-8 flex justify-center gap-4">
          <Link href="/assessment">
            <Button className="bg-white text-medical-blue border-2 border-white hover:bg-medical-blue hover:text-white px-6 py-2 font-semibold shadow-lg">
              Start New Case
            </Button>
          </Link>
          <Button
            variant="outline"
            className="bg-white/20 border-2 border-white text-white hover:bg-white hover:text-medical-blue px-6 py-2 font-semibold shadow-lg"
            onClick={() => window.location.href = "/api/logout"}
          >
            Log Out
          </Button>
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
                      <div className="flex items-center text-gray-600">
                        <User className="h-4 w-4 mr-2" />
                        {assessment.model}
                      </div>
                      {assessment.classification?.woundType && (
                        <div className="flex items-center text-gray-600">
                          <MapPin className="h-4 w-4 mr-2" />
                          {assessment.classification.woundType}
                        </div>
                      )}
                    </div>

                    {/* View Care Plan Button */}
                    <div className="mt-4">
                      <Link href={`/care-plan/${assessment.caseId}`}>
                        <Button size="sm" className="w-full bg-medical-blue hover:bg-medical-blue/90">
                          <FileText className="h-4 w-4 mr-2" />
                          View Care Plan
                        </Button>
                      </Link>
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