import { callOpenAI } from './openai';
import { callGemini } from './gemini';
import { storage } from '../storage';
import { InsertAgentQuestion } from '@shared/schema';

export async function analyzeAssessmentForQuestions(
  sessionId: string,
  contextData: any
): Promise<any[]> {
  const { imageAnalysis, audience, model } = contextData;
  
  // Check confidence level - if high confidence (>0.75), don't ask questions
  const confidence = imageAnalysis.confidence || 0.5;
  if (confidence > 0.75) {
    console.log(`High confidence (${confidence}) - skipping questions`);
    return [];
  }
  
  console.log(`Low confidence (${confidence}) - generating questions`);
  
  const analysisPrompt = `
You are an AI wound care specialist. Based on the image analysis, determine if you need to ask the user questions to improve diagnostic confidence.

WOUND ANALYSIS RESULTS:
${JSON.stringify(imageAnalysis, null, 2)}

TARGET AUDIENCE: ${audience}

DIAGNOSTIC CONFIDENCE ASSESSMENT:
1. If the wound type confidence is HIGH (>75%) and key characteristics are clear: Return empty array []
2. If the wound type confidence is MEDIUM-LOW (≤75%) or diagnosis is uncertain: Generate 2-3 targeted questions WITHOUT answers

RESPONSE FORMAT:
Return a JSON array of objects with this structure:
[
  {
    "id": "q1",
    "question": "Specific diagnostic question to clarify uncertainty",
    "answer": "",
    "category": "category_name",
    "confidence": 0.0
  }
]

QUESTION CATEGORIES:
- "wound_history" - How did this wound occur? When did it start?
- "pain_level" - Current pain and discomfort levels
- "medical_history" - Relevant medical conditions affecting healing
- "current_care" - Current treatment and care routine
- "symptoms" - Associated symptoms and recent changes
- "mobility" - Impact on daily activities and positioning
- "support" - Available care support at home

GUIDELINES FOR UNCERTAIN DIAGNOSES:
- Ask specific questions that help differentiate between possible wound types
- Focus on history, symptoms, and context not visible in the image
- Leave answer field empty - user will fill these in
- Ask maximum 3 questions per round
- Prioritize questions that most impact treatment decisions
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

    // Clean the response to remove any markdown formatting
    const cleanedResponse = response
      .replace(/```json/g, '')
      .replace(/```/g, '')
      .trim();
      
    const questions = JSON.parse(cleanedResponse);
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