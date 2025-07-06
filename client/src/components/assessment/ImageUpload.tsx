import { useState, useEffect } from "react";
import { Camera, Upload, ArrowRight, RefreshCw, X, Plus, Info } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { Alert, AlertDescription } from "@/components/ui/alert";
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
        totalImages: state.selectedImages.length
      });
      
      return await assessmentApi.initialAnalysis(
        primaryImage,
        state.audience,
        model,
        additionalImages
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
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Step 2: Upload Wound Images</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          
          {/* Helpful Guidance */}
          <Alert className="bg-blue-50 border-blue-200">
            <Info className="h-4 w-4 text-blue-600" />
            <AlertDescription className="text-blue-800">
              <strong>For best analysis results, upload multiple images:</strong>
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
                    
                    {/* Image Controls */}
                    <div className="absolute top-2 right-2 flex space-x-1">
                      {state.selectedImage !== image.file && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setPrimaryImage(image.id)}
                          className="bg-white/90 border-gray-300 text-xs px-2 py-1"
                        >
                          Set Primary
                        </Button>
                      )}
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => removeImage(image.id)}
                        className="bg-white/90 border-gray-300 text-red-600 hover:bg-red-50"
                      >
                        <X className="h-3 w-3" />
                      </Button>
                    </div>
                    
                    {/* Image Description */}
                    <div className="p-2 bg-gray-50 text-xs text-gray-600">
                      {index === 0 && "Primary image for AI analysis"}
                      {index === 1 && "Additional context"}
                      {index === 2 && "Different angle"}
                      {index === 3 && "Close-up details"}
                      {index === 4 && "Scale reference"}
                    </div>
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
                <span className="text-sm font-medium text-gray-700">Analyzing wound images...</span>
                <span className="text-sm text-gray-500">{elapsedTime}s</span>
              </div>
              <Progress value={progress} className="w-full" />
              <div className="text-xs text-gray-500 space-y-1">
                <div>• Processing {state.selectedImages.length} image{state.selectedImages.length !== 1 ? 's' : ''}</div>
                <div>• Estimated time: {getEstimatedTime(state.model || 'gemini-2.5-pro')}s</div>
                {state.model?.includes('gemini') && (
                  <div>• Gemini models provide detailed medical analysis (may take longer)</div>
                )}
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