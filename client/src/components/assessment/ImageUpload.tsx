import { useState, useEffect } from "react";
import { Camera, Upload, ArrowRight, RefreshCw } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import type { StepProps } from "./shared/AssessmentTypes";
import { assessmentApi, assessmentHelpers } from "./shared/AssessmentUtils";
import type { WoundClassification } from "@/../../shared/schema";

export default function ImageUpload({ state, onStateChange, onNextStep }: StepProps) {
  const { toast } = useToast();
  const [progress, setProgress] = useState(0);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [startTime, setStartTime] = useState<number | null>(null);

  // Get estimated processing time based on model
  const getEstimatedTime = (model: string): number => {
    if (model.includes('gemini')) {
      return 65; // 60 seconds + 5 second buffer
    } else if (model.includes('gpt-4o')) {
      return 25; // GPT-4o is usually faster
    } else {
      return 20; // GPT-3.5 variants
    }
  };



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
      
      // Ensure model is set with fallback
      const model = state.model || 'gemini-2.5-pro';
      
      console.log('Frontend - sending analysis request with:', {
        audience: state.audience,
        model: model,
        imageFile: state.selectedImage?.name
      });
      
      return await assessmentApi.initialAnalysis(
        state.selectedImage,
        state.audience,
        model
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

  // Progress bar animation and timer
  useEffect(() => {
    let interval: NodeJS.Timeout;
    let timerInterval: NodeJS.Timeout;
    
    if (initialAnalysisMutation.isPending) {
      if (!startTime) {
        const now = Date.now();
        setStartTime(now);
        setProgress(0);
        setElapsedTime(0);
      }
      
      const estimatedTime = getEstimatedTime(state.model || 'gemini-2.5-pro');
      
      // Update progress bar
      interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 95) return prev; // Don't complete until actually done
          const timeElapsed = (Date.now() - (startTime || Date.now())) / 1000;
          const expectedProgress = (timeElapsed / estimatedTime) * 100;
          return Math.min(expectedProgress + Math.random() * 10, 95);
        });
      }, 500);
      
      // Update elapsed time counter
      timerInterval = setInterval(() => {
        if (startTime) {
          setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
        }
      }, 1000);
    } else if (initialAnalysisMutation.isSuccess) {
      setProgress(100);
    } else {
      setProgress(0);
      setElapsedTime(0);
      setStartTime(null);
    }

    return () => {
      if (interval) clearInterval(interval);
      if (timerInterval) clearInterval(timerInterval);
    };
  }, [initialAnalysisMutation.isPending, initialAnalysisMutation.isSuccess, startTime, state.model]);

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
              {state.woundClassification && (
                <div className="mt-4 p-3 bg-gray-50 rounded-lg border">
                  <div className="text-sm font-medium text-gray-700 mb-2">Analysis Complete</div>
                  <div className="space-y-1 text-xs text-gray-600">
                    {state.woundClassification.confidence && (
                      <div><strong>AI Confidence:</strong> {Math.round(state.woundClassification.confidence * 100)}%</div>
                    )}
                    <div><strong>Wound Type:</strong> {state.woundClassification.woundType}</div>
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
          <div className="space-y-4">
            {initialAnalysisMutation.isPending ? (
              <div className="space-y-4">
                <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                  <div className="flex items-center justify-center gap-2 mb-3">
                    <RefreshCw className="h-5 w-5 animate-spin text-medical-blue" />
                    <span className="font-medium text-medical-blue">Analyzing Image</span>
                  </div>
                  
                  <div className="space-y-3">
                    <Progress value={progress} className="w-full" />
                    
                    <div className="flex justify-between items-center text-sm text-gray-600">
                      <span>
                        {progress < 20 && "Processing image..."}
                        {progress >= 20 && progress < 40 && "Detecting wound boundaries..."}
                        {progress >= 40 && progress < 60 && "Analyzing wound characteristics..."}
                        {progress >= 60 && progress < 80 && "Generating diagnostic questions..."}
                        {progress >= 80 && "Finalizing analysis..."}
                      </span>
                      <span>
                        {elapsedTime}s / ~{getEstimatedTime(state.model || 'gemini-2.5-pro')}s
                      </span>
                    </div>
                    
                    {state.model?.includes('gemini') && (
                      <div className="text-xs text-blue-600 bg-blue-100 p-2 rounded">
                        <strong>Note:</strong> Gemini provides thorough medical analysis but may take up to 60 seconds.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <Button 
                onClick={handleStartAnalysis}
                className="w-full bg-medical-blue hover:bg-medical-blue/90"
              >
                Start AI Analysis
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
} 