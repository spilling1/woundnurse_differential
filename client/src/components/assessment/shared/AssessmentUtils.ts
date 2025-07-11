import { apiRequest } from "@/lib/queryClient";
import type { AIGeneratedQuestion, WoundClassification, AudienceType, ModelType } from "./AssessmentTypes";

export const assessmentApi = {
  // Step 1: Initial image analysis
  initialAnalysis: async (primaryImage: File, audience: AudienceType, model: ModelType, additionalImages?: File[], bodyRegion?: { id: string; name: string }) => {
    const formData = new FormData();
    
    // Add primary image first
    formData.append('images', primaryImage);
    
    // Add additional images if provided
    if (additionalImages && additionalImages.length > 0) {
      additionalImages.forEach(image => {
        formData.append('images', image);
      });
    }
    
    formData.append('audience', audience);
    formData.append('model', model);
    formData.append('analysisType', 'initial');
    
    // Add body region information if provided
    if (bodyRegion) {
      formData.append('bodyRegion', JSON.stringify(bodyRegion));
    }
    
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
    userFeedback: string,
    existingCaseId?: string | null,
    forceNew?: boolean,
    bodyRegion?: { id: string; name: string }
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
    
    if (existingCaseId) {
      formData.append('existingCaseId', existingCaseId);
    }
    if (forceNew) {
      formData.append('forceNew', 'true');
    }
    if (bodyRegion) {
      formData.append('bodyRegion', JSON.stringify(bodyRegion));
    }

    const response = await fetch('/api/assessment/final-plan', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
      },
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
    round: number,
    bodyRegion?: { id: string; name: string }
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
    if (bodyRegion) {
      formData.append('bodyRegion', JSON.stringify(bodyRegion));
    }

    const response = await fetch('/api/assessment/follow-up-questions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
      },
      body: formData
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `Failed to generate follow-up questions (${response.status})`);
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
    { value: 'family' as const, label: 'Caregiver', desc: 'Simple, step-by-step guidance' },
    { value: 'medical' as const, label: 'Medical Professional', desc: 'Clinical terminology and protocols' }
  ],

  // Get model option configurations from API
  getModelOptions: async () => {
    try {
      const response = await fetch('/api/ai-analysis-models');
      if (!response.ok) {
        throw new Error('Failed to fetch AI models');
      }
      const models = await response.json();
      return models.map((model: any) => ({
        value: model.modelId,
        label: model.displayName + (model.isDefault ? ' (Default)' : ''),
        isDefault: model.isDefault
      }));
    } catch (error) {
      console.error('Error fetching AI models:', error);
      // Fallback to hardcoded options if API fails
      return [
        { value: 'gemini-2.5-pro' as const, label: 'Gemini 2.5 Pro (Recommended)', isDefault: true },
        { value: 'gpt-4o' as const, label: 'GPT-4o', isDefault: false },
        { value: 'gpt-3.5' as const, label: 'GPT-3.5', isDefault: false },
        { value: 'gpt-3.5-pro' as const, label: 'GPT-3.5 Pro', isDefault: false }
      ];
    }
  },

  // Get default model from API
  getDefaultModel: async () => {
    try {
      const response = await fetch('/api/ai-analysis-models');
      if (!response.ok) {
        throw new Error('Failed to fetch AI models');
      }
      const models = await response.json();
      const defaultModel = models.find((model: any) => model.isDefault);
      return defaultModel ? defaultModel.modelId : 'gemini-2.5-pro';
    } catch (error) {
      console.error('Error fetching default model:', error);
      return 'gemini-2.5-pro';
    }
  }
}; 