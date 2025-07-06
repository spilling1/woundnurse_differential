import { useState, useEffect } from "react";
import { CheckCircle } from "lucide-react";
import type { AssessmentFlowState, FlowStep } from "./shared/AssessmentTypes";
import { assessmentHelpers } from "./shared/AssessmentUtils";
import AudienceSelection from "./AudienceSelection";
import ImageUpload from "./ImageUpload";
import AIQuestions from "./AIQuestions";
import CarePlanGeneration from "./CarePlanGeneration";


export default function AssessmentFlow() {
  // Main assessment flow state
  const [state, setState] = useState<AssessmentFlowState>({
    currentStep: 'audience',
    audience: 'family',
    model: 'gemini-2.5-pro', // Default fallback
    selectedImage: null,
    imagePreview: null,
    selectedImages: [],
    aiQuestions: [],
    woundClassification: null,
    finalCaseId: null,
    userFeedback: '',
    selectedAlternative: null,
    questionRound: 1,
    answeredQuestions: []
  });

  // Load default model from API on mount
  useEffect(() => {
    const loadDefaultModel = async () => {
      try {
        const defaultModel = await assessmentHelpers.getDefaultModel();
        setState(prev => ({ ...prev, model: defaultModel }));
      } catch (error) {
        console.error('Failed to load default model:', error);
        // Keep fallback default
      }
    };
    
    loadDefaultModel();
  }, []);

  // Update state function
  const handleStateChange = (updates: Partial<AssessmentFlowState>) => {
    console.log('AssessmentFlow: State change requested:', updates);
    console.log('AssessmentFlow: Current model before update:', state.model);
    setState(prev => {
      const newState = { ...prev, ...updates };
      // Ensure model is never set to null accidentally
      if (newState.model === null || newState.model === undefined) {
        console.warn('AssessmentFlow: Model was null/undefined, preserving previous value:', prev.model);
        newState.model = prev.model || 'gemini-2.5-pro';
      }
      console.log('AssessmentFlow: New state model:', newState.model);
      return newState;
    });
  };

  // Navigation functions
  const handleNextStep = () => {
    const steps: FlowStep[] = ['audience', 'upload', 'ai-questions', 'generating-plan', 'final-plan'];
    const currentIndex = steps.indexOf(state.currentStep);
    if (currentIndex < steps.length - 1) {
      const nextStep = steps[currentIndex + 1];
      handleStateChange({ currentStep: nextStep });
    }
  };

  const handlePrevStep = () => {
    const steps: FlowStep[] = ['audience', 'upload', 'ai-questions', 'generating-plan', 'final-plan'];
    const currentIndex = steps.indexOf(state.currentStep);
    if (currentIndex > 0) {
      const prevStep = steps[currentIndex - 1];
      handleStateChange({ currentStep: prevStep });
    }
  };



  // Render step content
  const renderStepContent = () => {
    const stepProps = {
      state,
      onStateChange: handleStateChange,
      onNextStep: handleNextStep,
      onPrevStep: handlePrevStep
    };

    switch (state.currentStep) {
      case 'audience':
        return <AudienceSelection {...stepProps} />;
      case 'upload':
        return <ImageUpload {...stepProps} />;
      case 'ai-questions':
        return <AIQuestions {...stepProps} />;
      case 'generating-plan':
        return <CarePlanGeneration {...stepProps} />;
      case 'final-plan':
        return (
          <div className="text-center py-8">
            <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Assessment Complete!</h3>
            <p className="text-gray-600">Your care plan has been generated and saved.</p>
          </div>
        );
      default:
        return null;
    }
  };

  // Progress indicator steps
  const getProgressSteps = () => {
    const steps = [
      { key: 'audience', label: 'Audience' },
      { key: 'upload', label: 'Upload' },
      { key: 'ai-questions', label: 'AI Analysis' },
      { key: 'generating-plan', label: 'Generating Plan' },
      { key: 'final-plan', label: 'Complete' }
    ];

    return steps.map((step, index) => {
      const isActive = state.currentStep === step.key;
      const isCompleted = steps.findIndex(s => s.key === state.currentStep) > index;
      
      return (
        <div key={step.key} className="flex items-center">
          <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
            isCompleted ? 'bg-green-500 text-white' :
            isActive ? 'bg-medical-blue text-white' : 
            'bg-gray-200 text-gray-600'
          }`}>
            {isCompleted ? <CheckCircle className="h-4 w-4" /> : index + 1}
          </div>
          <div className="ml-2 text-sm font-medium text-gray-600">
            {step.label}
          </div>
          {index < steps.length - 1 && (
            <div className={`w-16 h-1 mx-4 ${
              isCompleted ? 'bg-green-500' : 'bg-gray-200'
            }`} />
          )}
        </div>
      );
    });
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Progress indicator */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          {getProgressSteps()}
        </div>
      </div>

      {/* Step content */}
      {renderStepContent()}
    </div>
  );
} 