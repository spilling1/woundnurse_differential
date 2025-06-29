import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { ArrowLeft, Save, FileText, AlertTriangle, CheckCircle, RotateCcw } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { useMutation, useQuery } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";

export default function AgentsPage() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const [content, setContent] = useState("");
  const [hasChanges, setHasChanges] = useState(false);
  const [originalContent, setOriginalContent] = useState("");

  // Fetch current Agents.md content
  const { data: agentsData, isLoading } = useQuery({
    queryKey: ['/api/agents'],
    queryFn: () => fetch('/api/agents').then(res => res.json()),
  });

  // Update content when data loads
  useEffect(() => {
    if (agentsData?.content) {
      setContent(agentsData.content);
      setOriginalContent(agentsData.content);
      setHasChanges(false);
    }
  }, [agentsData]);

  // Track changes
  useEffect(() => {
    setHasChanges(content !== originalContent);
  }, [content, originalContent]);

  // Save changes mutation
  const saveMutation = useMutation({
    mutationFn: async (newContent: string) => {
      return apiRequest('POST', '/api/agents', { content: newContent });
    },
    onSuccess: () => {
      toast({
        title: "Agents.md Updated",
        description: "The AI agent rules have been successfully updated.",
      });
      setOriginalContent(content);
      setHasChanges(false);
      queryClient.invalidateQueries({ queryKey: ['/api/agents'] });
    },
    onError: (error: any) => {
      toast({
        title: "Save Failed",
        description: error.message || "Failed to update Agents.md file.",
        variant: "destructive",
      });
    },
  });

  const handleSave = () => {
    if (content.trim() === "") {
      toast({
        title: "Invalid Content",
        description: "Agents.md cannot be empty.",
        variant: "destructive",
      });
      return;
    }
    saveMutation.mutate(content);
  };

  const handleReset = () => {
    setContent(originalContent);
    setHasChanges(false);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-bg-light flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-medical-blue mx-auto mb-3"></div>
          <p className="text-gray-600">Loading Agents.md...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-bg-light">
      {/* Header */}
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <Button 
                variant="ghost" 
                onClick={() => setLocation('/')}
                className="mr-4"
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Assessment
              </Button>
              <div className="flex items-center">
                <FileText className="text-medical-blue text-xl mr-3" />
                <h1 className="text-xl font-semibold text-gray-900">AI Agent Configuration</h1>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              {hasChanges && (
                <Badge variant="secondary" className="bg-yellow-100 text-yellow-800">
                  Unsaved Changes
                </Badge>
              )}
              <Button 
                variant="outline" 
                size="sm"
                onClick={handleReset}
                disabled={!hasChanges}
              >
                <RotateCcw className="mr-2 h-4 w-4" />
                Reset
              </Button>
              <Button 
                size="sm"
                onClick={handleSave}
                disabled={saveMutation.isPending || !hasChanges}
                className="bg-medical-blue hover:bg-blue-700"
              >
                <Save className="mr-2 h-4 w-4" />
                {saveMutation.isPending ? 'Saving...' : 'Save Changes'}
              </Button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Important Notice */}
        <Card className="mb-6">
          <CardContent className="p-6">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex items-start">
                <AlertTriangle className="text-blue-600 mr-3 mt-1 h-5 w-5" />
                <div>
                  <p className="text-sm text-blue-800 font-medium">AI Agent Instructions</p>
                  <p className="text-sm text-blue-700 mt-1">
                    These instructions guide the AI agent's behavior when analyzing wounds and generating care plans. 
                    Changes affect all future assessments. All case history is now stored securely in the database.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* File Statistics */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <Card>
            <CardContent className="p-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-medical-blue">{content.split('\n').length}</p>
                <p className="text-sm text-gray-600">Total Lines</p>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-medical-blue">{content.length}</p>
                <p className="text-sm text-gray-600">Characters</p>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-medical-blue">
                  {content.split('\n').filter(line => line.trim().startsWith('#')).length}
                </p>
                <p className="text-sm text-gray-600">Instruction Sections</p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Editor */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Agent Instructions</span>
              <div className="flex items-center space-x-2">
                {hasChanges ? (
                  <Badge variant="secondary" className="bg-yellow-100 text-yellow-800">
                    Modified
                  </Badge>
                ) : (
                  <Badge variant="secondary" className="bg-green-100 text-green-800">
                    <CheckCircle className="w-3 h-3 mr-1" />
                    Saved
                  </Badge>
                )}
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="border rounded-lg">
                <Textarea
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  rows={25}
                  className="font-mono text-sm border-0 resize-none"
                  placeholder="Enter AI agent instructions here..."
                />
              </div>
              
              <div className="flex justify-between items-center text-sm text-gray-500">
                <span>Use Markdown format for proper structuring</span>
                <span>{content.split(' ').length} words</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Instructions */}
        <Card className="mt-6">
          <CardHeader>
            <CardTitle>Editing Guidelines</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 text-sm text-gray-600">
              <p><strong>Structure:</strong> Keep the markdown format with proper headers (##) for case organization.</p>
              <p><strong>Cases:</strong> Each case should include wound details, classification, and care plan results.</p>
              <p><strong>Rules:</strong> Include specific guidelines for wound analysis and care plan generation.</p>
              <p><strong>Safety:</strong> Always review changes carefully as this affects AI behavior across all assessments.</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}