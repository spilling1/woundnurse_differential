import { useState, useEffect } from "react";
import { useLocation, useParams } from "wouter";
import { ArrowLeft, Save, Star, FileText, AlertCircle, CheckCircle, RefreshCw, Image, AlertTriangle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useMutation, useQuery } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

export default function NurseEvaluation() {
  const [, setLocation] = useLocation();
  const params = useParams();
  const { toast } = useToast();
  
  const [editedCarePlan, setEditedCarePlan] = useState("");
  const [rating, setRating] = useState("");
  const [nurseNotes, setNurseNotes] = useState("");
  const [additionalInstructions, setAdditionalInstructions] = useState("");
  const [selectedWoundType, setSelectedWoundType] = useState("");
  const [overrideWoundType, setOverrideWoundType] = useState(false);
  const [medicalHelpNeeded, setMedicalHelpNeeded] = useState(false);
  const [editableContext, setEditableContext] = useState<any>({});
  const [clinicalSummary, setClinicalSummary] = useState<any>({});
  const [hasChanges, setHasChanges] = useState(false);
  const [isRerunning, setIsRerunning] = useState(false);

  // Extract case ID from URL params
  const caseId = params.caseId;

  const { data: assessmentData, isLoading } = useQuery({
    queryKey: ['/api/assessment', caseId],
    enabled: !!caseId,
    queryFn: () => fetch(`/api/assessment/${caseId}`).then(res => res.json()),
  });

  useEffect(() => {
    if (!assessmentData) return;
    
    if (assessmentData.carePlan) {
      setEditedCarePlan(assessmentData.carePlan);
    }
    if (assessmentData.classification?.woundType) {
      setSelectedWoundType(assessmentData.classification.woundType);
    }
    
    // Initialize editable context data from the assessment
    let contextData: any = {};
    try {
      if (assessmentData.contextData) {
        contextData = typeof assessmentData.contextData === 'string' 
          ? JSON.parse(assessmentData.contextData)
          : assessmentData.contextData;
      }
    } catch (e) {
      console.warn('Failed to parse contextData:', e);
      contextData = {};
    }
    
    const initialContext = {
      woundOrigin: assessmentData.woundOrigin || contextData.woundOrigin || '',
      medicalHistory: assessmentData.medicalHistory || contextData.medicalHistory || '',
      woundChanges: assessmentData.woundChanges || contextData.woundChanges || '',
      currentCare: assessmentData.currentCare || contextData.currentCare || '',
      woundPain: assessmentData.woundPain || contextData.woundPain || '',
      supportAtHome: assessmentData.supportAtHome || contextData.supportAtHome || '',
      mobilityStatus: assessmentData.mobilityStatus || contextData.mobilityStatus || '',
      nutritionStatus: assessmentData.nutritionStatus || contextData.nutritionStatus || '',
    };
    setEditableContext(initialContext);

    // Initialize clinical summary from classification data
    const classification = typeof assessmentData.classification === 'string' 
      ? JSON.parse(assessmentData.classification)
      : assessmentData.classification || {};
    
    const initialClinicalSummary = {
      location: classification.location || '',
      size: classification.size || '',
      stage: classification.stage || '',
      woundBed: classification.woundBed || '',
      exudateLevel: classification.exudateLevel || classification.exudate || '',
      signsOfInfection: classification.signsOfInfection || classification.infection || '',
      additionalObservations: classification.additionalObservations || classification.observations || ''
    };
    setClinicalSummary(initialClinicalSummary);
  }, [assessmentData]);

  const saveEvaluationMutation = useMutation({
    mutationFn: async (data: any) => {
      const response = await apiRequest('POST', '/api/nurse-evaluation', data);
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Evaluation Saved",
        description: "Nurse evaluation and instructions have been updated successfully.",
      });
      setHasChanges(false);
    },
    onError: (error: any) => {
      toast({
        title: "Save Failed",
        description: error.message || "Failed to save evaluation.",
        variant: "destructive",
      });
    },
  });

  const rerunEvaluationMutation = useMutation({
    mutationFn: async (data: any) => {
      const response = await apiRequest('POST', '/api/nurse-rerun-evaluation', data);
      return response.json();
    },
    onSuccess: (result: any) => {
      console.log('Re-run result:', result);
      if (result.carePlan) {
        setEditedCarePlan(result.carePlan);
      }
      setIsRerunning(false);
      toast({
        title: "Care Plan Refreshed",
        description: "The care plan has been updated with your modifications.",
      });
      // Invalidate and refetch the assessment data
      queryClient.invalidateQueries({ queryKey: ['/api/assessment', caseId] });
      queryClient.refetchQueries({ queryKey: ['/api/assessment', caseId] });
    },
    onError: (error: any) => {
      setIsRerunning(false);
      toast({
        title: "Re-run Failed",
        description: error.message || "Failed to re-run evaluation.",
        variant: "destructive",
      });
    },
  });

  const updateAgentInstructionsMutation = useMutation({
    mutationFn: async (instructions: string) => {
      const response = await apiRequest('POST', '/api/agents/add-instructions', { instructions });
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Instructions Added",
        description: "Additional instructions have been added to the agent guidelines.",
      });
      setAdditionalInstructions("");
    },
    onError: (error: any) => {
      toast({
        title: "Update Failed", 
        description: error.message || "Failed to update agent instructions.",
        variant: "destructive",
      });
    }
  });

  const handleSave = async () => {
    if (!caseId) return;

    try {
      // Save nurse evaluation
      await saveEvaluationMutation.mutateAsync({
        caseId,
        editedCarePlan,
        rating,
        nurseNotes,
        medicalHelpNeeded
      });
    } catch (error) {
      console.error('Save error:', error);
    }
  };

  const handleRerunEvaluation = () => {
    if (!caseId) return;
    
    setIsRerunning(true);
    rerunEvaluationMutation.mutate({
      caseId,
      woundType: overrideWoundType ? selectedWoundType : null,
      contextData: editableContext,
      clinicalSummary: clinicalSummary
    });
  };

  const handleAddInstructions = () => {
    if (!additionalInstructions.trim()) return;
    updateAgentInstructionsMutation.mutate(additionalInstructions);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-bg-light flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-medical-blue mx-auto mb-3"></div>
          <p className="text-gray-600">Loading assessment...</p>
        </div>
      </div>
    );
  }

  if (!assessmentData) {
    return (
      <div className="min-h-screen bg-bg-light flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-600 mb-4">Assessment not found</p>
          <Button onClick={() => setLocation('/')}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Assessment
          </Button>
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
                onClick={() => setLocation(`/care-plan/${caseId}`)}
                className="mr-4"
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Care Plan
              </Button>
              <div className="flex items-center">
                <FileText className="text-medical-blue text-xl mr-3" />
                <h1 className="text-xl font-semibold text-gray-900">Nurse Evaluation</h1>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <Badge variant="secondary">Case: {caseId}</Badge>
              <Button 
                onClick={handleSave}
                disabled={saveEvaluationMutation.isPending}
                className="bg-medical-blue hover:bg-blue-700"
              >
                <Save className="mr-2 h-4 w-4" />
                {saveEvaluationMutation.isPending ? 'Saving...' : 'Save Evaluation'}
              </Button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Assessment Overview */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Assessment Overview</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <span className="font-medium text-gray-700">Wound Type:</span>
                <span className="ml-2 text-gray-600">{assessmentData.classification?.woundType}</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">Stage:</span>
                <span className="ml-2 text-gray-600">{assessmentData.classification?.stage}</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">Model Used:</span>
                <span className="ml-2 text-gray-600">{assessmentData.model}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column */}
          <div className="space-y-6">
            {/* Wound Image */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Image className="mr-2 h-5 w-5" />
                  Wound Image
                </CardTitle>
              </CardHeader>
              <CardContent>
                {assessmentData?.imageData ? (
                  <div className="text-center">
                    <img
                      src={`data:${assessmentData.imageMimeType || 'image/jpeg'};base64,${assessmentData.imageData}`}
                      alt="Wound assessment"
                      className="max-w-full h-auto rounded-lg border border-gray-200 shadow-sm"
                      style={{ maxHeight: '400px' }}
                    />
                  </div>
                ) : (
                  <p className="text-gray-500 text-center py-8">No image available</p>
                )}
              </CardContent>
            </Card>

            {/* Re-run Evaluation - All in One Panel */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <RefreshCw className="mr-2 h-5 w-5" />
                  Re-run Evaluation
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {/* Wound Type Override */}
                  <div className="space-y-3">
                    <div className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id="override-wound-type"
                        checked={overrideWoundType}
                        onChange={(e) => setOverrideWoundType(e.target.checked)}
                        className="rounded border-gray-300"
                      />
                      <Label htmlFor="override-wound-type">Override AI wound type classification</Label>
                    </div>
                    
                    {overrideWoundType && (
                      <div>
                        <Label htmlFor="wound-type">Wound Type Override</Label>
                        <Select value={selectedWoundType} onValueChange={setSelectedWoundType}>
                          <SelectTrigger>
                            <SelectValue placeholder="Select wound type" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="pressure-ulcer">Pressure Ulcer</SelectItem>
                            <SelectItem value="diabetic-ulcer">Diabetic Ulcer</SelectItem>
                            <SelectItem value="venous-ulcer">Venous Ulcer</SelectItem>
                            <SelectItem value="arterial-ulcer">Arterial Ulcer</SelectItem>
                            <SelectItem value="surgical-wound">Surgical Wound</SelectItem>
                            <SelectItem value="traumatic-wound">Traumatic Wound</SelectItem>
                            <SelectItem value="laceration">Laceration</SelectItem>
                            <SelectItem value="abrasion">Abrasion</SelectItem>
                            <SelectItem value="burn">Burn</SelectItem>
                            <SelectItem value="other">Other</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    )}
                  </div>

                  {/* Clinical Assessment Summary */}
                  <div className="border rounded-lg p-4 bg-gray-50">
                    <Label className="font-medium mb-3 block">Clinical Assessment Summary</Label>
                    <div className="space-y-3">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <div>
                          <Label htmlFor="location">Location</Label>
                          <Input
                            id="location"
                            value={clinicalSummary.location || ''}
                            onChange={(e) => {
                              setClinicalSummary({...clinicalSummary, location: e.target.value});
                              setHasChanges(true);
                            }}
                            placeholder="arm"
                          />
                        </div>
                        <div>
                          <Label htmlFor="size">Size</Label>
                          <Input
                            id="size"
                            value={clinicalSummary.size || ''}
                            onChange={(e) => {
                              setClinicalSummary({...clinicalSummary, size: e.target.value});
                              setHasChanges(true);
                            }}
                            placeholder="medium"
                          />
                        </div>
                        <div>
                          <Label htmlFor="stage">Stage</Label>
                          <Input
                            id="stage"
                            value={clinicalSummary.stage || ''}
                            onChange={(e) => {
                              setClinicalSummary({...clinicalSummary, stage: e.target.value});
                              setHasChanges(true);
                            }}
                            placeholder="N/A"
                          />
                        </div>
                        <div>
                          <Label htmlFor="wound-bed">Wound Bed</Label>
                          <Input
                            id="wound-bed"
                            value={clinicalSummary.woundBed || ''}
                            onChange={(e) => {
                              setClinicalSummary({...clinicalSummary, woundBed: e.target.value});
                              setHasChanges(true);
                            }}
                            placeholder="granulating"
                          />
                        </div>
                        <div>
                          <Label htmlFor="exudate-level">Exudate Level</Label>
                          <Input
                            id="exudate-level"
                            value={clinicalSummary.exudateLevel || ''}
                            onChange={(e) => {
                              setClinicalSummary({...clinicalSummary, exudateLevel: e.target.value});
                              setHasChanges(true);
                            }}
                            placeholder="low"
                          />
                        </div>
                      </div>
                      <div>
                        <Label htmlFor="signs-infection">Signs of Infection</Label>
                        <Textarea
                          id="signs-infection"
                          value={clinicalSummary.signsOfInfection || ''}
                          onChange={(e) => {
                            setClinicalSummary({...clinicalSummary, signsOfInfection: e.target.value});
                            setHasChanges(true);
                          }}
                          rows={2}
                          placeholder="Redness, warmth, odor, etc."
                        />
                      </div>
                      <div>
                        <Label htmlFor="additional-observations">Additional Observations</Label>
                        <Textarea
                          id="additional-observations"
                          value={clinicalSummary.additionalObservations || ''}
                          onChange={(e) => {
                            setClinicalSummary({...clinicalSummary, additionalObservations: e.target.value});
                            setHasChanges(true);
                          }}
                          rows={3}
                          placeholder="Additional clinical observations, periwound skin condition, etc."
                        />
                      </div>
                    </div>
                  </div>

                  {/* Medical Alert Checkbox */}
                  <div className="flex items-center space-x-3 mb-4">
                    <input
                      type="checkbox"
                      id="medical-help-needed"
                      checked={medicalHelpNeeded}
                      onChange={(e) => {
                        setMedicalHelpNeeded(e.target.checked);
                        setHasChanges(true);
                      }}
                      className="rounded border-gray-300 text-red-600 focus:ring-red-500"
                    />
                    <Label htmlFor="medical-help-needed" className="text-sm font-medium text-red-600">
                      Medical Help Needed - This case requires immediate medical attention
                    </Label>
                  </div>
                  {medicalHelpNeeded && (
                    <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                      <p className="text-sm text-red-800">
                        <strong>Alert:</strong> This case has been flagged for urgent medical review. 
                        The patient should seek immediate medical attention from a qualified healthcare provider.
                      </p>
                    </div>
                  )}

                  {/* Additional Agent Instructions */}
                  <div className="border rounded-lg p-4 bg-blue-50">
                    <Label className="font-medium mb-3 block">Additional Agent Instructions</Label>
                    <div className="space-y-3">
                      <Textarea
                        value={additionalInstructions}
                        onChange={(e) => setAdditionalInstructions(e.target.value)}
                        rows={3}
                        placeholder="Example: When assessing foot wounds, always ask about diabetes status regardless of visual appearance..."
                      />
                      <Button 
                        onClick={handleAddInstructions}
                        disabled={!additionalInstructions.trim() || updateAgentInstructionsMutation.isPending}
                        className="w-full"
                        variant="secondary"
                        size="sm"
                      >
                        <Save className="mr-2 h-4 w-4" />
                        {updateAgentInstructionsMutation.isPending ? 'Adding...' : 'Add to Agent Instructions'}
                      </Button>
                    </div>
                  </div>
                  
                  {/* Re-run Button at Bottom */}
                  <Button 
                    onClick={handleRerunEvaluation}
                    disabled={isRerunning || rerunEvaluationMutation.isPending || (overrideWoundType && !selectedWoundType)}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white"
                  >
                    <RefreshCw className={`mr-2 h-4 w-4 ${(isRerunning || rerunEvaluationMutation.isPending) ? 'animate-spin' : ''}`} />
                    {isRerunning || rerunEvaluationMutation.isPending ? 'Refreshing...' : 'Refresh Care Plan'}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Care Plan Editing */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Care Plan Review & Edit</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <Label>Generated Care Plan</Label>
                  <Textarea
                    value={editedCarePlan}
                    onChange={(e) => {
                      setEditedCarePlan(e.target.value);
                      setHasChanges(true);
                    }}
                    rows={25}
                    className="font-mono text-sm"
                    placeholder="Review and edit the AI-generated care plan..."
                  />
                </div>
              </CardContent>
            </Card>

            {/* Patient Context Review */}
            <Card>
              <CardHeader>
                <CardTitle>Patient Context Review</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="grid grid-cols-1 gap-3">
                    <div>
                      <Label htmlFor="wound-origin" className="text-sm">Wound Origin</Label>
                      <Input
                        id="wound-origin"
                        value={editableContext.woundOrigin || ''}
                        onChange={(e) => {
                          setEditableContext({...editableContext, woundOrigin: e.target.value});
                          setHasChanges(true);
                        }}
                        placeholder="How did the wound occur?"
                        className="min-w-0 w-full"
                        style={{ 
                          width: `${Math.max(200, (editableContext.woundOrigin || '').length * 8 + 50)}px` 
                        }}
                      />
                    </div>
                    <div>
                      <Label htmlFor="medical-history" className="text-sm">Medical History</Label>
                      <Input
                        id="medical-history"
                        value={editableContext.medicalHistory || ''}
                        onChange={(e) => {
                          setEditableContext({...editableContext, medicalHistory: e.target.value});
                          setHasChanges(true);
                        }}
                        placeholder="Relevant medical conditions"
                        className="min-w-0 w-full"
                        style={{ 
                          width: `${Math.max(200, (editableContext.medicalHistory || '').length * 8 + 50)}px` 
                        }}
                      />
                    </div>
                    <div>
                      <Label htmlFor="current-care" className="text-sm">Current Care</Label>
                      <Input
                        id="current-care"
                        value={editableContext.currentCare || ''}
                        onChange={(e) => {
                          setEditableContext({...editableContext, currentCare: e.target.value});
                          setHasChanges(true);
                        }}
                        placeholder="Current treatment regimen"
                        className="min-w-0 w-full"
                        style={{ 
                          width: `${Math.max(200, (editableContext.currentCare || '').length * 8 + 50)}px` 
                        }}
                      />
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Rating */}
            <Card>
              <CardHeader>
                <CardTitle>Care Plan Quality Rating</CardTitle>
              </CardHeader>
              <CardContent>
                <RadioGroup value={rating} onValueChange={(value) => {
                  setRating(value);
                  setHasChanges(true);
                }}>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="excellent" id="excellent" />
                    <Label htmlFor="excellent" className="flex items-center">
                      <Star className="w-4 h-4 text-yellow-500 mr-1" />
                      Excellent - No changes needed
                    </Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="good" id="good" />
                    <Label htmlFor="good" className="flex items-center">
                      <CheckCircle className="w-4 h-4 text-green-500 mr-1" />
                      Good - Minor adjustments made
                    </Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="needs-improvement" id="needs-improvement" />
                    <Label htmlFor="needs-improvement" className="flex items-center">
                      <AlertCircle className="w-4 h-4 text-orange-500 mr-1" />
                      Needs Improvement - Significant changes required
                    </Label>
                  </div>
                </RadioGroup>
              </CardContent>
            </Card>

            {/* Nurse Notes */}
            <Card>
              <CardHeader>
                <CardTitle>Professional Notes</CardTitle>
              </CardHeader>
              <CardContent>
                <Textarea
                  value={nurseNotes}
                  onChange={(e) => {
                    setNurseNotes(e.target.value);
                    setHasChanges(true);
                  }}
                  rows={6}
                  placeholder="Add your professional observations, recommendations for AI improvement, or specific case notes..."
                />
              </CardContent>
            </Card>


          </div>


        </div>

        {hasChanges && (
          <Card className="mt-6">
            <CardContent className="p-4">
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <p className="text-sm text-yellow-800">
                  <strong>Unsaved Changes:</strong> You have unsaved changes. Click "Save Evaluation" to persist your updates.
                </p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}