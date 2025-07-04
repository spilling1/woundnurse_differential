import { ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import type { StepProps } from "./shared/AssessmentTypes";
import { assessmentHelpers } from "./shared/AssessmentUtils";

export default function AudienceSelection({ state, onStateChange, onNextStep }: StepProps) {
  const audienceOptions = assessmentHelpers.getAudienceOptions();
  const modelOptions = assessmentHelpers.getModelOptions();

  const handleAudienceChange = (audience: typeof state.audience) => {
    onStateChange({ audience });
  };

  const handleModelChange = (model: typeof state.model) => {
    onStateChange({ model });
  };

  const handleContinue = () => {
    onStateChange({ currentStep: 'upload' });
    onNextStep();
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Step 1: Select Your Audience</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-gray-600">Who will be using this care plan?</p>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {audienceOptions.map(option => (
            <div
              key={option.value}
              className={`p-4 border rounded-lg cursor-pointer transition-all ${
                state.audience === option.value 
                  ? 'border-medical-blue bg-blue-50' 
                  : 'border-gray-200 hover:border-gray-300'
              }`}
              onClick={() => handleAudienceChange(option.value)}
            >
              <div className="font-medium">{option.label}</div>
              <div className="text-sm text-gray-600">{option.desc}</div>
            </div>
          ))}
        </div>
        
        <div className="mt-6">
          <Label>AI Model</Label>
          <select 
            value={state.model} 
            onChange={(e) => handleModelChange(e.target.value as typeof state.model)}
            className="w-full mt-1 p-2 border rounded-md"
          >
            {modelOptions.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        
        <Button 
          onClick={handleContinue}
          className="w-full bg-medical-blue hover:bg-medical-blue/90"
        >
          Continue to Image Upload
          <ArrowRight className="ml-2 h-4 w-4" />
        </Button>
      </CardContent>
    </Card>
  );
} 