import { apiRequest } from "@/lib/queryClient";
import type { AIGeneratedQuestion, WoundClassification, AudienceType, ModelType } from "./AssessmentTypes";

export const assessmentApi = {
  // Step 1: Initial image analysis
  initialAnalysis: async (image: File, audience: AudienceType, model: ModelType) => {
    const formData = new FormData();
    formData.append('image', image);
    formData.append('audience', audience);
    formData.append('model', model);
    formData.append('analysisType', 'initial');
    
    const response = await apiRequest('POST', '/api/assessment/initial-analysis', formData);
    return await response.json();
  },

  // Step 2: Generate preliminary care plan
  preliminaryPlan: async (
    image: File | null,
    audience: AudienceType,
    model: ModelType,
    questions: AIGeneratedQuestion[],
    classification: WoundClassification | null,
    selectedAlternative: string | null,
    userFeedback: string
  ) => {
    const response = await apiRequest('POST', '/api/assessment/preliminary-plan', {
      imageData: image,
      audience,
      model,
      questions,
      classification,
      selectedAlternative,
      userFeedback
    });
    return await response.json();
  },

  // Generate final care plan directly after questions
  finalPlan: async (
    image: File | null,
    audience: AudienceType,
    model: ModelType,
    questions: AIGeneratedQuestion[],
    classification: WoundClassification | null,
    userFeedback: string
  ) => {
    const formData = new FormData();
    if (image) {
      formData.append('image', image);
    }
    formData.append('audience', audience);
    formData.append('model', model);
    formData.append('questions', JSON.stringify(questions));
    formData.append('classification', JSON.stringify(classification));
    formData.append('userFeedback', userFeedback);

    const response = await fetch('/api/assessment/final-plan', {
      method: 'POST',
      body: formData,
      credentials: 'include',
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  },

  // Generate feedback questions
  feedbackQuestions: async (
    classification: WoundClassification | null,
    userFeedback: string,
    audience: AudienceType,
    model: ModelType
  ) => {
    const response = await apiRequest('POST', '/api/assessment/feedback-questions', {
      classification,
      userFeedback,
      audience,
      model
    });
    return await response.json();
  },

  // Generate follow-up questions
  followUpQuestions: async (
    image: File | null,
    audience: AudienceType,
    model: ModelType,
    previousQuestions: AIGeneratedQuestion[],
    classification: WoundClassification | null,
    round: number
  ) => {
    const formData = new FormData();
    if (image) {
      formData.append('image', image);
    }
    formData.append('audience', audience);
    formData.append('model', model);
    formData.append('previousQuestions', JSON.stringify(previousQuestions));
    formData.append('classification', JSON.stringify(classification));
    formData.append('round', round.toString());

    const response = await fetch('/api/assessment/follow-up-questions', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error('Failed to generate follow-up questions');
    }

    return await response.json();
  }
};

export const assessmentHelpers = {
  // Handle image file selection
  handleImageSelect: (
    file: File,
    setImagePreview: (preview: string) => void
  ) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  },

  // Update AI question answer
  updateQuestionAnswer: (
    questions: AIGeneratedQuestion[],
    questionId: string,
    newAnswer: string
  ): AIGeneratedQuestion[] => {
    return questions.map(q => 
      q.id === questionId ? { ...q, answer: newAnswer } : q
    );
  },

  // Calculate overall confidence score
  calculateOverallConfidence: (questions: AIGeneratedQuestion[]): number => {
    if (questions.length === 0) return 1.0;
    const totalConfidence = questions.reduce((sum, q) => sum + q.confidence, 0);
    return totalConfidence / questions.length;
  },

  // Check if all questions are answered
  areAllQuestionsAnswered: (questions: AIGeneratedQuestion[]): boolean => {
    return questions.every(q => q.answer && q.answer.trim() !== '');
  },

  // Get audience option configurations
  getAudienceOptions: () => [
    { value: 'family' as const, label: 'Family Caregiver', desc: 'Simple, step-by-step guidance' },
    { value: 'patient' as const, label: 'Patient', desc: 'Self-care focused instructions' },
    { value: 'medical' as const, label: 'Medical Professional', desc: 'Clinical terminology and protocols' }
  ],

  // Get model option configurations
  getModelOptions: () => [
    { value: 'gemini-2.5-pro' as const, label: 'Gemini 2.5 Pro (Recommended)' },
    { value: 'gemini-2.5-flash' as const, label: 'Gemini 2.5 Flash' },
    { value: 'gpt-4o' as const, label: 'GPT-4o' },
    { value: 'gpt-3.5' as const, label: 'GPT-3.5' },
    { value: 'gpt-3.5-pro' as const, label: 'GPT-3.5 Pro' }
  ]
}; 