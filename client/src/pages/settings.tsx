import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { ArrowLeft, Settings, Save, RefreshCw } from "lucide-react";
import { useLocation } from "wouter";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

export default function SettingsPage() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const [instructions, setInstructions] = useState("");

  // Fetch current AI instructions
  const { data: agentData, isLoading } = useQuery({
    queryKey: ["/api/agents"],
    retry: false,
  });

  // Initialize instructions when data loads
  useState(() => {
    if (agentData?.content) {
      setInstructions(agentData.content);
    }
  }, [agentData]);

  // Mutation to update AI instructions
  const updateMutation = useMutation({
    mutationFn: async (newInstructions: string) => {
      const response = await apiRequest('/api/agents', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content: newInstructions }),
      });
      return response;
    },
    onSuccess: () => {
      toast({
        title: "Settings Updated",
        description: "AI configuration has been successfully updated.",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/agents"] });
    },
    onError: (error: Error) => {
      toast({
        title: "Update Failed",
        description: error.message || "Failed to update AI configuration.",
        variant: "destructive",
      });
    },
  });

  const handleSave = () => {
    updateMutation.mutate(instructions);
  };

  const handleReset = () => {
    if (agentData?.content) {
      setInstructions(agentData.content);
      toast({
        title: "Reset Complete",
        description: "Settings have been reset to saved values.",
      });
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header Navigation */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setLocation('/my-cases')}
                className="mr-4"
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to My Cases
              </Button>
              <div className="flex items-center">
                <Settings className="text-medical-blue text-xl mr-3" />
                <div>
                  <h1 className="text-lg font-semibold text-gray-900">Settings</h1>
                  <p className="text-sm text-gray-500">AI Configuration Management</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Settings className="h-5 w-5 mr-2 text-medical-blue" />
              AI Configuration
            </CardTitle>
            <p className="text-sm text-gray-600">
              Configure the AI agent instructions that guide wound assessment and care plan generation. 
              Changes will affect all future assessments.
            </p>
          </CardHeader>
          <CardContent className="space-y-6">
            {isLoading ? (
              <div className="flex items-center justify-center p-8">
                <RefreshCw className="h-6 w-6 animate-spin text-medical-blue mr-2" />
                <span className="text-gray-600">Loading configuration...</span>
              </div>
            ) : (
              <>
                <div className="space-y-2">
                  <Label htmlFor="instructions" className="text-sm font-medium">
                    AI Agent Instructions
                  </Label>
                  <Textarea
                    id="instructions"
                    value={instructions}
                    onChange={(e) => setInstructions(e.target.value)}
                    placeholder="Enter AI agent instructions for wound care assessment..."
                    className="min-h-[400px] font-mono text-sm"
                    disabled={updateMutation.isPending}
                  />
                  <p className="text-xs text-gray-500">
                    These instructions guide the AI in analyzing wound images and generating care plans. 
                    Use clear, specific language for best results.
                  </p>
                </div>

                <div className="flex justify-between items-center pt-4 border-t">
                  <div className="text-sm text-gray-500">
                    {agentData?.updatedAt && (
                      <span>Last updated: {new Date(agentData.updatedAt).toLocaleString()}</span>
                    )}
                  </div>
                  <div className="flex space-x-3">
                    <Button
                      variant="outline"
                      onClick={handleReset}
                      disabled={updateMutation.isPending || !agentData?.content}
                    >
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Reset
                    </Button>
                    <Button
                      onClick={handleSave}
                      disabled={updateMutation.isPending || instructions === agentData?.content}
                      className="bg-medical-blue hover:bg-blue-700"
                    >
                      {updateMutation.isPending ? (
                        <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <Save className="h-4 w-4 mr-2" />
                      )}
                      Save Changes
                    </Button>
                  </div>
                </div>
              </>
            )}
          </CardContent>
        </Card>

        {/* Information Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Best Practices</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-600 space-y-2">
              <ul className="list-disc list-inside space-y-1">
                <li>Use clear, specific medical terminology</li>
                <li>Include evidence-based care protocols</li>
                <li>Specify different approaches for various wound types</li>
                <li>Consider patient safety and professional standards</li>
              </ul>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Impact</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-gray-600 space-y-2">
              <ul className="list-disc list-inside space-y-1">
                <li>Changes affect all new assessments</li>
                <li>Existing assessments remain unchanged</li>
                <li>Updates are logged for audit purposes</li>
                <li>Regular review is recommended</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}