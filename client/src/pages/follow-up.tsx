import { useState, useRef } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useParams, useLocation } from "wouter";
import { useAuth } from "@/hooks/useAuth";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { ArrowLeft, Upload, Camera, Clock, TrendingUp, AlertCircle, Loader2 } from "lucide-react";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

const followUpSchema = z.object({
  audience: z.enum(['family', 'patient', 'medical']),
  model: z.enum(['gpt-4o', 'gpt-3.5', 'gpt-3.5-pro', 'gemini-2.5-flash', 'gemini-2.5-pro']),
  progressNotes: z.string().min(10, "Please provide at least 10 characters describing the progress"),
  treatmentResponse: z.string().min(10, "Please provide at least 10 characters describing treatment response"),
  woundOrigin: z.string().optional(),
  medicalHistory: z.string().optional(),
  woundChanges: z.string().optional(),
  currentCare: z.string().optional(),
  woundPain: z.string().optional(),
  supportAtHome: z.string().optional(),
  mobilityStatus: z.string().optional(),
  nutritionStatus: z.string().optional(),
});

export default function FollowUpAssessment() {
  const { caseId } = useParams();
  const [, setLocation] = useLocation();
  const { isAuthenticated } = useAuth();
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const form = useForm<z.infer<typeof followUpSchema>>({
    resolver: zodResolver(followUpSchema),
    defaultValues: {
      audience: 'patient',
      model: 'gpt-4o',
      progressNotes: '',
      treatmentResponse: '',
      woundOrigin: '',
      medicalHistory: '',
      woundChanges: '',
      currentCare: '',
      woundPain: '',
      supportAtHome: '',
      mobilityStatus: '',
      nutritionStatus: '',
    },
  });

  // Get original case data
  const { data: originalCase, isLoading: isLoadingCase } = useQuery({
    queryKey: [`/api/assessment/${caseId}`],
    enabled: !!caseId,
  });

  const followUpMutation = useMutation({
    mutationFn: async (formData: FormData) => {
      const response = await fetch(`/api/follow-up/${caseId}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to submit follow-up assessment');
      }

      return response.json();
    },
    onSuccess: (data) => {
      toast({
        title: "Follow-up Assessment Complete",
        description: `Successfully created version ${data.version} of your care plan.`,
      });
      
      // Invalidate related queries
      queryClient.invalidateQueries({ queryKey: ['/api/my-cases'] });
      queryClient.invalidateQueries({ queryKey: ['/api/assessment', caseId] });
      
      // Navigate to the updated care plan
      setLocation(`/care-plan/${caseId}`);
    },
    onError: (error: Error) => {
      toast({
        title: "Assessment Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const handleFileSelect = (file: File) => {
    if (file) {
      if (file.size > 10 * 1024 * 1024) {
        toast({
          title: "File Too Large",
          description: "Please select an image under 10MB.",
          variant: "destructive",
        });
        return;
      }

      if (!file.type.match(/^image\/(jpeg|jpg|png)$/)) {
        toast({
          title: "Invalid File Type",
          description: "Please select a JPEG or PNG image.",
          variant: "destructive",
        });
        return;
      }

      setSelectedFile(file);
      
      // Create preview URL
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const onSubmit = async (values: z.infer<typeof followUpSchema>) => {
    if (!selectedFile) {
      toast({
        title: "Image Required",
        description: "Please upload a current wound image for comparison.",
        variant: "destructive",
      });
      return;
    }

    const formData = new FormData();
    formData.append('image', selectedFile);
    
    Object.entries(values).forEach(([key, value]) => {
      if (value !== undefined && value !== '') {
        formData.append(key, value.toString());
      }
    });

    followUpMutation.mutate(formData);
  };

  if (isLoadingCase) {
    return (
      <div className="min-h-screen bg-bg-light flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  if (!originalCase) {
    return (
      <div className="min-h-screen bg-bg-light flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="h-16 w-16 text-red-500 mx-auto mb-4" />
          <h1 className="text-2xl font-bold text-gray-900 mb-2">Case Not Found</h1>
          <p className="text-gray-600 mb-6">The original case could not be found.</p>
          <Button onClick={() => setLocation('/my-cases')}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to My Cases
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-bg-light">
      {/* Header */}
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <Button 
                variant="ghost" 
                onClick={() => setLocation('/my-cases')}
                className="mr-4"
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Cases
              </Button>
              <div className="flex items-center">
                <TrendingUp className="text-medical-blue text-xl mr-3" />
                <h1 className="text-xl font-semibold text-gray-900">Follow-up Assessment</h1>
              </div>
            </div>
            <div className="flex items-center">
              <Badge variant="secondary">Case: {caseId}</Badge>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Original Case Summary */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Clock className="h-5 w-5 mr-2 text-medical-blue" />
              Original Assessment Summary
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="text-sm text-gray-600">
                  <strong>Date:</strong> {new Date(originalCase.createdAt).toLocaleDateString()}
                </div>
                <div className="text-sm text-gray-600">
                  <strong>Wound Type:</strong> {originalCase.classification?.woundType || 'Not specified'}
                </div>
                <div className="text-sm text-gray-600">
                  <strong>Audience:</strong> {originalCase.audience}
                </div>
              </div>
              <div className="space-y-2">
                <div className="text-sm text-gray-600">
                  <strong>AI Model:</strong> {originalCase.model}
                </div>
                <div className="text-sm text-gray-600">
                  <strong>Stage:</strong> {originalCase.classification?.stage || 'Not applicable'}
                </div>
                <div className="text-sm text-gray-600">
                  <strong>Location:</strong> {originalCase.classification?.location || 'Not specified'}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
            
            {/* Progress Assessment */}
            <Card>
              <CardHeader>
                <CardTitle>Progress Assessment</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                
                {/* Progress Notes */}
                <FormField
                  control={form.control}
                  name="progressNotes"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Progress Since Last Assessment *</FormLabel>
                      <FormControl>
                        <Textarea
                          placeholder="Describe how the wound has changed since the last assessment. Include observations about healing, pain levels, mobility improvements, etc."
                          rows={4}
                          {...field}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                {/* Treatment Response */}
                <FormField
                  control={form.control}
                  name="treatmentResponse"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Treatment Response *</FormLabel>
                      <FormControl>
                        <Textarea
                          placeholder="How has the wound responded to the previous treatment plan? What treatments have been most/least effective?"
                          rows={4}
                          {...field}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

              </CardContent>
            </Card>

            {/* Current Wound Image */}
            <Card>
              <CardHeader>
                <CardTitle>Current Wound Image</CardTitle>
              </CardHeader>
              <CardContent>
                <div
                  className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                    dragActive ? 'border-medical-blue bg-blue-50' : 'border-gray-300'
                  }`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                >
                  {previewUrl ? (
                    <div className="space-y-4">
                      <img
                        src={previewUrl}
                        alt="Wound preview"
                        className="max-w-xs max-h-64 mx-auto rounded-lg border border-gray-200"
                      />
                      <div className="text-sm text-gray-600">
                        {selectedFile?.name} ({(selectedFile?.size || 0 / 1024).toFixed(1)} KB)
                      </div>
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => {
                          setSelectedFile(null);
                          setPreviewUrl(null);
                        }}
                      >
                        Remove Image
                      </Button>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <Upload className="h-12 w-12 text-gray-400 mx-auto" />
                      <div>
                        <h3 className="text-lg font-medium text-gray-900">Upload Current Wound Image</h3>
                        <p className="text-gray-600">
                          Upload a recent image of the wound for comparison with the previous assessment
                        </p>
                      </div>
                      <div className="flex flex-col sm:flex-row gap-3 justify-center">
                        <Button
                          type="button"
                          variant="outline"
                          onClick={() => fileInputRef.current?.click()}
                        >
                          <Camera className="mr-2 h-4 w-4" />
                          Choose File
                        </Button>
                      </div>
                      <p className="text-sm text-gray-500">
                        Supports JPEG and PNG files up to 10MB
                      </p>
                    </div>
                  )}
                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/jpg,image/png"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      handleFileSelect(e.target.files[0]);
                    }
                  }}
                />
              </CardContent>
            </Card>

            {/* Assessment Configuration */}
            <Card>
              <CardHeader>
                <CardTitle>Assessment Configuration</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                
                {/* Audience Selection */}
                <FormField
                  control={form.control}
                  name="audience"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Target Audience</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select audience" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="family">Family Caregivers</SelectItem>
                          <SelectItem value="patient">Patients</SelectItem>
                          <SelectItem value="medical">Medical Professionals</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                {/* AI Model Selection */}
                <FormField
                  control={form.control}
                  name="model"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>AI Model</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select AI model" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="gpt-4o">GPT-4o (Recommended)</SelectItem>
                          <SelectItem value="gpt-3.5">GPT-3.5</SelectItem>
                          <SelectItem value="gpt-3.5-pro">GPT-3.5 Pro</SelectItem>
                          <SelectItem value="gemini-2.5-flash">Gemini 2.5 Flash</SelectItem>
                          <SelectItem value="gemini-2.5-pro">Gemini 2.5 Pro</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />

              </CardContent>
            </Card>

            {/* Submit Button */}
            <div className="flex justify-end space-x-4">
              <Button
                type="button"
                variant="outline"
                onClick={() => setLocation('/my-cases')}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={followUpMutation.isPending || !selectedFile}
                className="bg-medical-blue hover:bg-medical-blue/90"
              >
                {followUpMutation.isPending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <TrendingUp className="mr-2 h-4 w-4" />
                    Generate Updated Care Plan
                  </>
                )}
              </Button>
            </div>

          </form>
        </Form>
      </div>
    </div>
  );
}