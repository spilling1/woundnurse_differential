import { useState, useEffect } from "react";
import { CheckCircle } from "lucide-react";
import type { AssessmentFlowState, FlowStep } from "./shared/AssessmentTypes";
import AudienceSelection from "./AudienceSelection";
import ImageUpload from "./ImageUpload";
import AIQuestions from "./AIQuestions";
import PreliminaryPlan from "./PreliminaryPlan";

export default function AssessmentFlow() {
  // Main assessment flow state
  const [state, setState] = useState<AssessmentFlowState>({
    currentStep: 'audience',
    audience: 'family',
    model: 'gpt-4o',
    selectedImage: null,
    imagePreview: null,
    aiQuestions: [],
    woundClassification: null,
    preliminaryPlan: null,
    finalCaseId: null,
    userFeedback: '',
    selectedAlternative: null,
    questionRound: 1,
    answeredQuestions: []
  });

  // Update state function
  const handleStateChange = (updates: Partial<AssessmentFlowState>) => {
    setState(prev => ({ ...prev, ...updates }));
  };

  // Navigation functions
  const handleNextStep = () => {
    const steps: FlowStep[] = ['audience', 'upload', 'ai-questions', 'preliminary-plan', 'final-plan'];
    const currentIndex = steps.indexOf(state.currentStep);
    if (currentIndex < steps.length - 1) {
      const nextStep = steps[currentIndex + 1];
      handleStateChange({ currentStep: nextStep });
    }
  };

  const handlePrevStep = () => {
    const steps: FlowStep[] = ['audience', 'upload', 'ai-questions', 'preliminary-plan', 'final-plan'];
    const currentIndex = steps.indexOf(state.currentStep);
    if (currentIndex > 0) {
      const prevStep = steps[currentIndex - 1];
      handleStateChange({ currentStep: prevStep });
    }
  };

  // Auto-trigger preliminary plan when reaching that step with no pending questions
  useEffect(() => {
    if (state.currentStep === 'preliminary-plan' && !state.preliminaryPlan && state.aiQuestions.length === 0) {
      console.log('Auto-triggering preliminary plan generation');
      // This would be handled by the PreliminaryPlan component
    }
  }, [state.currentStep, state.preliminaryPlan, state.aiQuestions.length]);

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
      case 'preliminary-plan':
        return <PreliminaryPlan {...stepProps} />;
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
      { key: 'preliminary-plan', label: 'Preliminary Plan' },
      { key: 'final-plan', label: 'Final Plan' }
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