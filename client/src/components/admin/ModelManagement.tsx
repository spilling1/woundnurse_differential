import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Brain, Edit, Trash2, Plus, Star } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface DetectionModel {
  id: number;
  name: string;
  description: string;
  enabled: boolean;
  priority: number;
  accuracy: number;
  speed: string;
}

interface AiAnalysisModel {
  id: number;
  name: string;
  displayName: string;
  description: string;
  enabled: boolean;
  isDefault: boolean;
  provider: string;
  capabilities: string[];
}

interface ModelManagementProps {
  detectionModels: DetectionModel[];
  aiModels: AiAnalysisModel[];
  isLoading: boolean;
}

export function ModelManagement({ detectionModels, aiModels, isLoading }: ModelManagementProps) {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const toggleDetectionModel = useMutation({
    mutationFn: async ({ id, enabled }: { id: number; enabled: boolean }) => {
      const response = await fetch(`/api/admin/detection-models/${id}/toggle`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ enabled }),
      });
      if (!response.ok) throw new Error('Failed to toggle model');
      return response.json();
    },
    onSuccess: () => {
      toast({ title: "Success", description: "Detection model updated successfully" });
      queryClient.invalidateQueries({ queryKey: ['/api/admin/detection-models'] });
    },
    onError: (error: any) => {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    }
  });

  const toggleAiModel = useMutation({
    mutationFn: async ({ id, enabled }: { id: number; enabled: boolean }) => {
      const response = await fetch(`/api/admin/ai-analysis-models/${id}/toggle`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ enabled }),
      });
      if (!response.ok) throw new Error('Failed to toggle model');
      return response.json();
    },
    onSuccess: () => {
      toast({ title: "Success", description: "AI model updated successfully" });
      queryClient.invalidateQueries({ queryKey: ['/api/admin/ai-analysis-models'] });
    },
    onError: (error: any) => {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    }
  });

  const setDefaultModel = useMutation({
    mutationFn: async (id: number) => {
      const response = await fetch(`/api/admin/ai-analysis-models/${id}/set-default`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
      });
      if (!response.ok) throw new Error('Failed to set default model');
      return response.json();
    },
    onSuccess: () => {
      toast({ title: "Success", description: "Default model updated successfully" });
      queryClient.invalidateQueries({ queryKey: ['/api/admin/ai-analysis-models'] });
    },
    onError: (error: any) => {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    }
  });

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Loading Models...</CardTitle>
          </CardHeader>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Detection Models */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Detection Models
          </CardTitle>
          <CardDescription>
            Manage wound detection and measurement models
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {detectionModels.map((model) => (
              <div key={model.id} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <h4 className="font-medium">{model.name}</h4>
                    <Badge variant={model.enabled ? 'default' : 'secondary'}>
                      {model.enabled ? 'Active' : 'Disabled'}
                    </Badge>
                  </div>
                  <p className="text-sm text-gray-600">{model.description}</p>
                  <div className="flex items-center gap-4 text-xs text-gray-500">
                    <span>Accuracy: {model.accuracy}%</span>
                    <span>Speed: {model.speed}</span>
                    <span>Priority: {model.priority}</span>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Switch
                    checked={model.enabled}
                    onCheckedChange={(enabled) => 
                      toggleDetectionModel.mutate({ id: model.id, enabled })
                    }
                    disabled={toggleDetectionModel.isPending}
                  />
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* AI Analysis Models */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            AI Analysis Models
          </CardTitle>
          <CardDescription>
            Manage AI models for wound classification and care plan generation
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {aiModels.map((model) => (
              <div key={model.id} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="space-y-1 flex-1">
                  <div className="flex items-center gap-2">
                    <h4 className="font-medium">{model.displayName}</h4>
                    {model.isDefault && (
                      <Badge variant="default" className="bg-yellow-500">
                        <Star className="h-3 w-3 mr-1" />
                        Default
                      </Badge>
                    )}
                    <Badge variant={model.enabled ? 'default' : 'secondary'}>
                      {model.enabled ? 'Active' : 'Disabled'}
                    </Badge>
                    <Badge variant="outline">{model.provider}</Badge>
                  </div>
                  <p className="text-sm text-gray-600">{model.description}</p>
                  <div className="flex flex-wrap gap-1 mt-2">
                    {model.capabilities.map((capability, index) => (
                      <Badge key={index} variant="outline" className="text-xs">
                        {capability}
                      </Badge>
                    ))}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {!model.isDefault && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setDefaultModel.mutate(model.id)}
                      disabled={setDefaultModel.isPending || !model.enabled}
                    >
                      <Star className="h-4 w-4" />
                    </Button>
                  )}
                  <Switch
                    checked={model.enabled}
                    onCheckedChange={(enabled) => 
                      toggleAiModel.mutate({ id: model.id, enabled })
                    }
                    disabled={toggleAiModel.isPending}
                  />
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}