import { useState } from "react";
import { Camera, Upload, ArrowRight, RefreshCw } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import type { StepProps } from "./shared/AssessmentTypes";
import { assessmentApi, assessmentHelpers } from "./shared/AssessmentUtils";
import type { WoundClassification } from "@/../../shared/schema";

export default function ImageUpload({ state, onStateChange, onNextStep }: StepProps) {
  const { toast } = useToast();

  // Helper function to get user-friendly detection method names
  const getDetectionMethodName = (model: string): string => {
    switch (model) {
      case 'yolo8':
      case 'yolov8':
      case 'smart-yolo-yolo':
        return 'YOLO v8 Detection';
      case 'smart-yolo-color':
      case 'color-detection':
        return 'Color-based Detection';
      case 'google-cloud-vision':
        return 'Google Cloud Vision';
      case 'azure-computer-vision':
        return 'Azure Computer Vision';
      case 'enhanced-fallback':
        return 'Enhanced Image Analysis';
      default:
        return 'Image Analysis';
    }
  };

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
              
              {/* Detection Method Information */}
              {state.woundClassification?.detectionMetadata && (
                <div className="mt-4 p-3 bg-gray-50 rounded-lg border">
                  <div className="text-sm font-medium text-gray-700 mb-2">Detection Analysis</div>
                  <div className="space-y-1 text-xs text-gray-600">
                    <div><strong>Detection Method:</strong> {getDetectionMethodName(state.woundClassification.detectionMetadata.model)}</div>
                    <div><strong>Classification Method:</strong> {state.woundClassification.classificationMethod || 'AI Vision'}</div>
                    {state.woundClassification.detectionMetadata.multipleWounds && (
                      <div><strong>Multiple Wounds:</strong> {state.woundClassification.detectionMetadata.multipleWounds ? 'Yes' : 'No'}</div>
                    )}
                    {state.woundClassification.confidence && (
                      <div><strong>AI Confidence:</strong> {Math.round(state.woundClassification.confidence * 100)}%</div>
                    )}
                  </div>
                </div>
              )}
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