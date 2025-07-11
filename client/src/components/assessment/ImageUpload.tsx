import { useState, useEffect } from "react";
import { Camera, Upload, ArrowRight, RefreshCw, X, Plus, Info } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { Alert, AlertDescription } from "@/components/ui/alert";
import type { StepProps } from "./shared/AssessmentTypes";
import { assessmentApi, assessmentHelpers } from "./shared/AssessmentUtils";
import type { WoundClassification } from "@/../../shared/schema";
import BodyRegionSelector from "./BodyRegionSelector";


export default function ImageUpload({ state, onStateChange, onNextStep }: StepProps) {
  const { toast } = useToast();
  const [, setLocation] = useLocation();
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

  // Handle multiple image file selection
  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    if (files.length === 0) return;

    // Limit to 5 images max
    const limitedFiles = files.slice(0, 5);
    const newImages = limitedFiles.map((file, index) => ({
      file,
      preview: URL.createObjectURL(file),
      id: `img_${Date.now()}_${index}`,
      description: ''
    }));

    // Set the first image as the primary image for backward compatibility
    if (newImages.length > 0) {
      onStateChange({ 
        selectedImage: newImages[0].file,
        imagePreview: newImages[0].preview,
        selectedImages: [...state.selectedImages, ...newImages]
      });
    }
  };

  // Remove an image
  const removeImage = (imageId: string) => {
    const updatedImages = state.selectedImages.filter(img => img.id !== imageId);
    onStateChange({ selectedImages: updatedImages });
    
    // If we removed the primary image, set a new one
    if (updatedImages.length > 0 && state.selectedImage) {
      const removedImage = state.selectedImages.find(img => img.id === imageId);
      if (removedImage && removedImage.file === state.selectedImage) {
        onStateChange({
          selectedImage: updatedImages[0].file,
          imagePreview: updatedImages[0].preview
        });
      }
    } else if (updatedImages.length === 0) {
      onStateChange({
        selectedImage: null,
        imagePreview: null
      });
    }
  };

  // Set primary image
  const setPrimaryImage = (imageId: string) => {
    const image = state.selectedImages.find(img => img.id === imageId);
    if (image) {
      onStateChange({
        selectedImage: image.file,
        imagePreview: image.preview
      });
    }
  };

  // Initial image analysis mutation
  const initialAnalysisMutation = useMutation({
    mutationFn: async () => {
      if (!state.selectedImage || state.selectedImages.length === 0) {
        throw new Error('No images selected');
      }
      
      // Ensure model is set with fallback
      const model = state.model || 'gemini-2.5-pro';
      
      // Get primary image and additional images
      const primaryImage = state.selectedImage;
      const additionalImages = state.selectedImages
        .filter(img => img.file !== primaryImage)
        .map(img => img.file);
      
      console.log('Frontend - sending analysis request with:', {
        audience: state.audience,
        model: model,
        primaryImage: primaryImage.name,
        additionalImages: additionalImages.length,
        totalImages: state.selectedImages.length,
        bodyRegion: state.bodyRegion
      });
      
      return await assessmentApi.initialAnalysis(
        primaryImage,
        state.audience,
        model,
        additionalImages,
        state.bodyRegion
      );
    },
    onSuccess: (data: any) => {
      console.log('ImageUpload: Analysis successful:', data);
      console.log('ImageUpload: Classification object:', data.classification);
      console.log('ImageUpload: Has unsupportedWoundType flag?', data.classification?.unsupportedWoundType);
      
      if (data.duplicateDetected) {
        // Handle duplicate detection at the beginning of the process
        onStateChange({
          duplicateInfo: data,
          currentStep: 'generating-plan' // Skip to plan generation to handle duplicate
        });
        // Don't call onNextStep() here since we're setting the step directly
      } else if (data.classification?.unsupportedWoundType) {
        // Handle unsupported wound type - redirect to unsupported wound page
        console.log('ImageUpload: Unsupported wound type detected, redirecting');
        const params = new URLSearchParams({
          woundType: data.classification.woundType || 'Unknown',
          confidence: ((data.classification.confidence || 0.85) * 100).toString(),
          reasoning: data.classification.reasoning || data.classification.additionalObservations || 'Visual analysis performed by AI',
          message: `This wound appears to be a ${data.classification.woundType} which is not currently supported by the Wound Nurse.`
        });
        
        // Add supported types if available
        if (data.classification.supportedTypes && Array.isArray(data.classification.supportedTypes)) {
          params.set('supportedTypes', data.classification.supportedTypes.join('|'));
        }
        
        console.log('ImageUpload: Redirecting to unsupported wound page with params:', params.toString());
        setLocation(`/unsupported-wound?${params.toString()}`);
      } else {
        // Normal flow - proceed with questions
        onStateChange({
          aiQuestions: data.questions || [],
          woundClassification: data.classification,
          currentStep: 'ai-questions'
        });
        onNextStep();
      }
    },
    onError: (error: any) => {
      console.log('ImageUpload error:', error);
      
      // Handle generic errors with simple toast message
      toast({
        title: "Analysis Failed",
        description: error.message || "An error occurred during analysis. Please try again.",
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

  const handleBodyRegionSelect = (regionId: string, regionName: string) => {
    onStateChange({
      bodyRegion: regionId ? { id: regionId, name: regionName } : undefined
    });
  };

  return (
    <div className="space-y-6">
      {/* Body Region Selection */}
      <BodyRegionSelector
        selectedRegion={state.bodyRegion?.id || null}
        onRegionSelect={handleBodyRegionSelect}
      />

      <Card>
        <CardHeader>
          <CardTitle>Step 2: Upload Wound Images</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          
          {/* Helpful Guidance */}
          <Alert className="bg-blue-50 border-blue-200">
            <Info className="h-4 w-4 text-blue-600" />
            <AlertDescription className="text-blue-800">
              <strong>The system can process with just one image, however for best analysis results, upload multiple images:</strong>
              <ul className="mt-2 ml-4 list-disc space-y-1 text-sm">
                <li><strong>Primary shot:</strong> Clear, well-lit photo of the entire wound</li>
                <li><strong>Close-up:</strong> Detailed view of wound bed and edges</li>
                <li><strong>Different angles:</strong> Side views to show depth</li>
                <li><strong>With scale:</strong> Include a coin or ruler for accurate measurements</li>
                <li><strong>Context:</strong> Wider view showing surrounding area</li>
              </ul>
            </AlertDescription>
          </Alert>

          {/* Upload Area */}
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
            <Camera className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600 mb-4">Upload wound images (up to 5 images)</p>
            
            <input
              type="file"
              accept="image/*"
              onChange={handleImageSelect}
              className="hidden"
              id="image-upload"
              multiple
            />
            <label htmlFor="image-upload">
              <Button variant="outline" asChild>
                <span>
                  <Upload className="mr-2 h-4 w-4" />
                  {state.selectedImages.length > 0 ? 'Add More Images' : 'Upload Images'}
                </span>
              </Button>
            </label>
          </div>

          {/* Image Gallery */}
          {state.selectedImages.length > 0 && (
            <div className="space-y-4">
              <h3 className="text-sm font-medium text-gray-700">
                Uploaded Images ({state.selectedImages.length})
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {state.selectedImages.map((image, index) => (
                  <div key={image.id} className="relative border rounded-lg overflow-hidden">
                    <img 
                      src={image.preview} 
                      alt={`Wound view ${index + 1}`}
                      className="w-full h-32 object-cover"
                    />
                    
                    {/* Primary Image Indicator */}
                    {state.selectedImage === image.file && (
                      <div className="absolute top-2 left-2 bg-green-500 text-white px-2 py-1 rounded text-xs font-medium">
                        Primary
                      </div>
                    )}
                    

                    

                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Analysis Results */}
          {state.woundClassification && (
            <div className="mt-4 p-3 bg-gray-50 rounded-lg border">
              <div className="text-sm font-medium text-gray-700 mb-2">Analysis Complete</div>
              <div className="space-y-1 text-xs text-gray-600">
                {state.woundClassification.confidence && (
                  <div><strong>AI Confidence:</strong> {Math.round(state.woundClassification.confidence * 100)}%</div>
                )}
                <div><strong>Wound Type:</strong> {state.woundClassification.woundType}</div>
                <div><strong>Images Analyzed:</strong> {state.selectedImages.length}</div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Analysis Progress */}
      {initialAnalysisMutation.isPending && (
        <Card>
          <CardContent className="pt-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <RefreshCw className="h-4 w-4 text-blue-500 animate-spin" />
                  <span className="text-sm font-medium text-blue-600">Analyzing Image</span>
                </div>
                <span className="text-sm text-gray-500">{elapsedTime}s / ~{getEstimatedTime(state.model || 'gemini-2.5-pro')}s</span>
              </div>
              <Progress value={progress} className="w-full" />
              
              {/* Detailed Step Messages */}
              <div className="text-sm text-gray-700 font-medium">
                {progress < 20 && "Processing image..."}
                {progress >= 20 && progress < 35 && "Detecting wound boundaries..."}
                {progress >= 35 && progress < 50 && "Analyzing wound characteristics..."}
                {progress >= 50 && progress < 65 && "Measuring wound dimensions..."}
                {progress >= 65 && progress < 80 && "Classifying wound type..."}
                {progress >= 80 && progress < 95 && "Generating assessment..."}
                {progress >= 95 && "Finalizing analysis..."}
              </div>
              
              <div className="text-xs text-blue-600 bg-blue-50 p-2 rounded">
                <strong>Note:</strong> Analysis can take up to {getEstimatedTime(state.model || 'gemini-2.5-pro')} seconds for thorough medical image processing. Please be patient while we generate your detailed assessment.
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Action Buttons */}
      <div className="flex justify-between">
        <Button variant="outline" onClick={() => window.history.back()}>
          Back
        </Button>
        
        <Button 
          onClick={handleStartAnalysis}
          disabled={state.selectedImages.length === 0 || initialAnalysisMutation.isPending}
          className="bg-medical-blue hover:bg-medical-blue/90"
        >
          {initialAnalysisMutation.isPending ? (
            <>
              <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <ArrowRight className="mr-2 h-4 w-4" />
              Start Analysis
            </>
          )}
        </Button>
      </div>
    </div>
  );
}