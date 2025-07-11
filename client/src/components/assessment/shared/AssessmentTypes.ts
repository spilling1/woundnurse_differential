export interface AIGeneratedQuestion {
  id: string;
  question: string;
  answer: string;
  category: string;
  confidence: number;
}

export interface WoundClassification {
  woundType: string;
  confidence: number;
  alternativeTypes: Array<{
    type: string;
    confidence: number;
    reasoning: string;
  }>;
}

export type FlowStep = 'audience' | 'upload' | 'ai-questions' | 'generating-plan' | 'final-plan';

export type AudienceType = 'family' | 'patient' | 'medical';

export type ModelType = 'gpt-4o' | 'gpt-3.5' | 'gpt-3.5-pro' | 'gemini-2.5-flash' | 'gemini-2.5-pro';

export interface ImageUpload {
  file: File;
  preview: string;
  id: string;
  description?: string;
}

export interface AssessmentFlowState {
  currentStep: FlowStep;
  audience: AudienceType;
  model: ModelType | null;
  selectedImage: File | null;
  imagePreview: string | null;
  selectedImages: ImageUpload[];
  aiQuestions: AIGeneratedQuestion[];
  woundClassification: WoundClassification | null;
  finalCaseId: string | null;
  userFeedback: string;
  selectedAlternative: string | null;
  questionRound: number;
  answeredQuestions: AIGeneratedQuestion[];
  duplicateInfo?: any;
  bodyRegion?: {
    id: string;
    name: string;
  };
  differentialRefinement?: any;
  showPage2Analysis?: boolean;
}

export interface StepProps {
  state: AssessmentFlowState;
  onStateChange: (updates: Partial<AssessmentFlowState>) => void;
  onNextStep: () => void;
  onPrevStep?: () => void;
} 