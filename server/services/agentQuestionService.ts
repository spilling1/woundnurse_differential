import { callOpenAI } from './openai';
import { callGemini } from './gemini';
import { storage } from '../storage';
import { InsertAgentQuestion } from '@shared/schema';

export async function analyzeAssessmentForQuestions(
  sessionId: string,
  contextData: any
): Promise<any[]> {
  const { imageAnalysis, audience, model } = contextData;
  
  const analysisPrompt = `
You are an AI wound care specialist. Based on the image analysis, generate 3-5 essential questions with AI-generated answers that would help create an optimal care plan.

WOUND ANALYSIS RESULTS:
${JSON.stringify(imageAnalysis, null, 2)}

TARGET AUDIENCE: ${audience}

TASK: Generate questions with AI-generated answers based on what you can observe or infer from the wound image analysis. These should be the most critical questions needed for creating a personalized care plan.

RESPONSE FORMAT:
Return a JSON array of objects with this structure:
[
  {
    "id": "q1",
    "question": "Essential question text",
    "answer": "AI-generated answer based on image analysis and medical knowledge",
    "category": "category_name",
    "confidence": 0.8
  }
]

QUESTION CATEGORIES:
- "wound_history" - How did this wound occur?
- "pain_level" - Current pain and discomfort levels
- "medical_history" - Relevant medical conditions
- "current_care" - Current treatment and care routine
- "symptoms" - Associated symptoms and changes
- "mobility" - Impact on daily activities
- "support" - Available care support

GUIDELINES:
- Generate realistic, informed answers based on visual analysis
- Focus on questions that significantly impact treatment decisions
- Tailor language complexity to the target audience
- Include confidence scores (0.6-0.9) based on how certain the AI analysis is
`;

  try {
    let response: string;
    
    if (model && model.startsWith('gemini')) {
      response = await callGemini(model, analysisPrompt);
    } else {
      const messages = [
        {
          role: "user",
          content: analysisPrompt
        }
      ];
      response = await callOpenAI(model || 'gpt-4o', messages);
    }

    const questions = JSON.parse(response);
    return Array.isArray(questions) ? questions : [];
    
  } catch (error) {
    console.error('Error generating AI questions:', error);
    return [];
  }
}

export async function getSessionQuestions(sessionId: string) {
  return await storage.getQuestionsBySession(sessionId);
}

export async function answerSessionQuestion(questionId: number, answer: string) {
  return await storage.answerQuestion(questionId, answer);
}

export async function checkSessionComplete(sessionId: string): Promise<boolean> {
  const unanswered = await storage.getUnansweredQuestions(sessionId);
  return unanswered.length === 0;
}