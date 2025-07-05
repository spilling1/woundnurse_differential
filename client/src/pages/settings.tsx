import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Settings, Save, RefreshCw, ArrowLeft } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "wouter";

function SettingsPage() {
  const [systemPrompts, setSystemPrompts] = useState("");
  const [carePlanStructure, setCarePlanStructure] = useState("");
  const [specificWoundCare, setSpecificWoundCare] = useState("");
  const [questionsGuidelines, setQuestionsGuidelines] = useState("");
  const { toast } = useToast();
  const queryClient = useQueryClient();

  // Query to fetch current AI instructions
  const { data: agentData, isLoading } = useQuery({
    queryKey: ["/api/agents"],
  });

  // Update textareas when data is loaded
  useEffect(() => {
    if (agentData && typeof agentData === 'object') {
      const data = agentData as any;
      setSystemPrompts(data.systemPrompts || "");
      setCarePlanStructure(data.carePlanStructure || "");
      setSpecificWoundCare(data.specificWoundCare || "");
      setQuestionsGuidelines(data.questionsGuidelines || "");
    }
  }, [agentData]);

  // Mutation to update AI instructions
  const updateMutation = useMutation({
    mutationFn: async (newInstructions: {
      systemPrompts: string;
      carePlanStructure: string;
      specificWoundCare: string;
      questionsGuidelines: string;
    }) => {
      const response = await fetch('/api/agents', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newInstructions),
      });
      if (!response.ok) {
        throw new Error('Failed to update instructions');
      }
      return response.json();
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
    updateMutation.mutate({
      systemPrompts,
      carePlanStructure,
      specificWoundCare,
      questionsGuidelines
    });
  };

  const handleReset = () => {
    if (agentData && typeof agentData === 'object') {
      const data = agentData as any;
      setSystemPrompts(data.systemPrompts || "");
      setCarePlanStructure(data.carePlanStructure || "");
      setSpecificWoundCare(data.specificWoundCare || "");
      setQuestionsGuidelines(data.questionsGuidelines || "");
      toast({
        title: "Reset Complete",
        description: "Settings have been reset to saved values.",
      });
    }
  };

  const hasChanges = () => {
    if (!agentData) return false;
    const data = agentData as any;
    return (
      systemPrompts !== (data.systemPrompts || "") ||
      carePlanStructure !== (data.carePlanStructure || "") ||
      specificWoundCare !== (data.specificWoundCare || "") ||
      questionsGuidelines !== (data.questionsGuidelines || "")
    );
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-bg-light flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-medical-blue mx-auto mb-3"></div>
          <p className="text-gray-600">Loading settings...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-bg-light">
      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-4">
            <Link href="/my-cases">
              <Button variant="outline" size="sm">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to My Cases
              </Button>
            </Link>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 flex items-center">
                <Settings className="mr-3 h-8 w-8 text-medical-blue" />
                AI Configuration Settings
              </h1>
              <p className="text-gray-600 mt-2">
                Configure how the AI analyzes wounds and generates care plans
              </p>
            </div>
          </div>
        </div>

        {/* Settings Content */}
        <Card className="shadow-lg border-0">
          <CardHeader className="bg-gradient-to-r from-medical-blue to-blue-600 text-white">
            <CardTitle className="text-xl font-semibold flex items-center">
              <Settings className="h-6 w-6 mr-3" />
              AI Instructions Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            <Tabs defaultValue="system" className="w-full">
              <TabsList className="grid w-full grid-cols-4 mb-6">
                <TabsTrigger value="system">System Prompts</TabsTrigger>
                <TabsTrigger value="structure">Care Plan Structure</TabsTrigger>
                <TabsTrigger value="wound">Specific Wound Care</TabsTrigger>
                <TabsTrigger value="questions">Questions Guidelines</TabsTrigger>
              </TabsList>

              <TabsContent value="system" className="space-y-4">
                <div>
                  <h3 className="text-lg font-semibold mb-2">System Prompts</h3>
                  <p className="text-gray-600 mb-4">
                    Core mission and behavior instructions for the AI assistant.
                  </p>
                  <Textarea
                    value={systemPrompts}
                    onChange={(e) => setSystemPrompts(e.target.value)}
                    rows={20}
                    className="font-mono text-sm"
                    placeholder="Enter system prompts..."
                  />
                </div>
              </TabsContent>

              <TabsContent value="structure" className="space-y-4">
                <div>
                  <h3 className="text-lg font-semibold mb-2">Care Plan Structure</h3>
                  <p className="text-gray-600 mb-4">
                    How the AI should format and organize care plan responses.
                  </p>
                  <Textarea
                    value={carePlanStructure}
                    onChange={(e) => setCarePlanStructure(e.target.value)}
                    rows={20}
                    className="font-mono text-sm"
                    placeholder="Enter care plan structure instructions..."
                  />
                </div>
              </TabsContent>

              <TabsContent value="wound" className="space-y-4">
                <div>
                  <h3 className="text-lg font-semibold mb-2">Specific Wound Care Instructions</h3>
                  <p className="text-gray-600 mb-4">
                    Medical knowledge and guidelines for different wound types.
                  </p>
                  <Textarea
                    value={specificWoundCare}
                    onChange={(e) => setSpecificWoundCare(e.target.value)}
                    rows={20}
                    className="font-mono text-sm"
                    placeholder="Enter specific wound care instructions..."
                  />
                </div>
              </TabsContent>

              <TabsContent value="questions" className="space-y-4">
                <div>
                  <h3 className="text-lg font-semibold mb-2">Questions Guidelines</h3>
                  <p className="text-gray-600 mb-4">
                    How to ask follow-up questions to improve diagnostic accuracy.
                  </p>
                  <Textarea
                    value={questionsGuidelines}
                    onChange={(e) => setQuestionsGuidelines(e.target.value)}
                    rows={20}
                    className="font-mono text-sm"
                    placeholder="Enter questions guidelines..."
                  />
                </div>
              </TabsContent>
            </Tabs>

            {/* Action Buttons */}
            <div className="flex justify-between items-center pt-6 border-t mt-6">
              <div className="text-sm text-gray-500">
                {agentData && typeof agentData === 'object' && 'lastModified' in agentData && (
                  <span>Last updated: {new Date((agentData as any).lastModified).toLocaleString()}</span>
                )}
              </div>
              <div className="flex space-x-3">
                <Button
                  variant="outline"
                  onClick={handleReset}
                  disabled={updateMutation.isPending || !hasChanges()}
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Reset
                </Button>
                <Button
                  onClick={handleSave}
                  disabled={updateMutation.isPending || !hasChanges()}
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
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default SettingsPage;