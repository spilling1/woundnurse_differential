import { useState } from "react";
import { Camera, Upload, ArrowRight, RefreshCw } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import type { StepProps } from "./shared/AssessmentTypes";
import { assessmentApi, assessmentHelpers } from "./shared/AssessmentUtils";

export default function ImageUpload({ state, onStateChange, onNextStep }: StepProps) {
  const { toast } = useToast();

  // Handle image file selection
  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onStateChange({ selectedImage: file });
      assessmentHelpers.handleImageSelect(file, (preview) => {
        onStateChange({ imagePreview: preview });
      });
    }
  };

  // Initial image analysis mutation
  const initialAnalysisMutation = useMutation({
    mutationFn: async () => {
      if (!state.selectedImage) throw new Error('No image selected');
      
      return await assessmentApi.initialAnalysis(
        state.selectedImage,
        state.audience,
        state.model
      );
    },
    onSuccess: (data: any) => {
      onStateChange({
        aiQuestions: data.questions || [],
        woundClassification: data.classification,
        currentStep: 'ai-questions'
      });
      onNextStep();
    },
    onError: (error: any) => {
      toast({
        title: "Analysis Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const handleStartAnalysis = () => {
    initialAnalysisMutation.mutate();
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Step 2: Upload Wound Image</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
          {state.imagePreview ? (
            <div className="space-y-4">
              <img 
                src={state.imagePreview} 
                alt="Wound preview" 
                className="max-w-full h-64 object-contain mx-auto rounded-lg"
              />
              <p className="text-sm text-gray-600">Click below to change image</p>
            </div>
          ) : (
            <div className="space-y-4">
              <Camera className="h-12 w-12 text-gray-400 mx-auto" />
              <p className="text-gray-600">Click to upload a wound image</p>
            </div>
          )}
          
          <input
            type="file"
            accept="image/*"
            onChange={handleImageSelect}
            className="hidden"
            id="image-upload"
          />
          <label htmlFor="image-upload">
            <Button variant="outline" className="mt-4" asChild>
              <span>
                <Upload className="mr-2 h-4 w-4" />
                {state.imagePreview ? 'Change Image' : 'Upload Image'}
              </span>
            </Button>
          </label>
        </div>
        
        {state.selectedImage && (
          <Button 
            onClick={handleStartAnalysis}
            disabled={initialAnalysisMutation.isPending}
            className="w-full bg-medical-blue hover:bg-medical-blue/90"
          >
            {initialAnalysisMutation.isPending ? (
              <>
                <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                Analyzing Image...
              </>
            ) : (
              <>
                Start AI Analysis
                <ArrowRight className="ml-2 h-4 w-4" />
              </>
            )}
          </Button>
        )}
      </CardContent>
    </Card>
  );
} 