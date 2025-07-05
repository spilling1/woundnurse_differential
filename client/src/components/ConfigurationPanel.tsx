import { Settings, Microscope } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { WoundContextData } from "./WoundQuestionnaire";

interface ConfigurationPanelProps {
  audience: 'family' | 'patient' | 'medical';
  model: 'gpt-4o' | 'gpt-3.5' | 'gpt-3.5-pro' | 'gemini-2.5-flash' | 'gemini-2.5-pro';
  onAudienceChange: (audience: 'family' | 'patient' | 'medical') => void;
  onModelChange: (model: 'gpt-4o' | 'gpt-3.5' | 'gpt-3.5-pro' | 'gemini-2.5-flash' | 'gemini-2.5-pro') => void;
  selectedFile: File | null;
  isProcessing: boolean;
  contextData?: WoundContextData;
  onStartAssessment: () => void;
  onAssessmentComplete: (data: any) => void;
}

export default function ConfigurationPanel({
  audience,
  model,
  onAudienceChange,
  onModelChange,
  selectedFile,
  isProcessing,
  contextData,
  onStartAssessment,
  onAssessmentComplete
}: ConfigurationPanelProps) {
  const { toast } = useToast();

  const assessmentMutation = useMutation({
    mutationFn: async () => {
      if (!selectedFile) {
        throw new Error('No image selected');
      }

      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('audience', audience);
      formData.append('model', model);
      
      // Add context data if available
      if (contextData) {
        Object.entries(contextData).forEach(([key, value]) => {
          if (value && value.trim()) {
            formData.append(key, value);
          }
        });
      }

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Assessment failed');
      }

      return response.json();
    },
    onSuccess: (data) => {
      onAssessmentComplete(data);
      toast({
        title: "Assessment Complete",
        description: "Wound analysis and care plan generated successfully.",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Assessment Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const handleStartAssessment = () => {
    onStartAssessment();
    assessmentMutation.mutate();
  };

  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex items-center mb-4">
          <Settings className="text-medical-blue text-lg mr-2" />
          <h2 className="text-lg font-semibold text-gray-900">Assessment Configuration</h2>
        </div>

        {/* Audience Selection */}
        <div className="mb-6">
          <Label className="text-sm font-medium text-gray-700 mb-3 block">Target Audience</Label>
          <RadioGroup value={audience} onValueChange={onAudienceChange}>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="family" id="family" />
              <Label htmlFor="family" className="text-sm">
                <span className="font-medium text-gray-900">Family Caregiver</span>
                <span className="block text-gray-500">Clear, step-by-step instructions</span>
              </Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="patient" id="patient" />
              <Label htmlFor="patient" className="text-sm">
                <span className="font-medium text-gray-900">Patient</span>
                <span className="block text-gray-500">Empowering self-care guidance</span>
              </Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="medical" id="medical" />
              <Label htmlFor="medical" className="text-sm">
                <span className="font-medium text-gray-900">Medical Professional</span>
                <span className="block text-gray-500">Clinical protocols and terminology</span>
              </Label>
            </div>
          </RadioGroup>
        </div>

        {/* Model Selection */}
        <div className="mb-6">
          <Label className="text-sm font-medium text-gray-700 mb-3 block">AI Model</Label>
          <Select value={model} onValueChange={onModelChange}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="gemini-2.5-pro">Gemini 2.5 Pro (Recommended)</SelectItem>
              <SelectItem value="gemini-2.5-flash">Gemini 2.5 Flash</SelectItem>
              <SelectItem value="gpt-4o">GPT-4o</SelectItem>
              <SelectItem value="gpt-3.5">GPT-3.5</SelectItem>
              <SelectItem value="gpt-3.5-pro">GPT-3.5 Pro</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-gray-500 mt-1">Higher models provide more detailed analysis</p>
        </div>

        {/* Action Button */}
        <Button 
          className="w-full bg-medical-blue hover:bg-blue-700"
          disabled={!selectedFile || isProcessing}
          onClick={handleStartAssessment}
        >
          <Microscope className="mr-2 h-4 w-4" />
          {isProcessing ? 'Processing...' : 'Start Assessment'}
        </Button>
      </CardContent>
    </Card>
  );
}
