import { callOpenAI } from './openai';
import { callGemini } from './gemini';
import { storage } from '../storage';
import { InsertAgentQuestion } from '@shared/schema';

export async function analyzeAssessmentForQuestions(
  sessionId: string,
  contextData: any
): Promise<any[]> {
  const { imageAnalysis, audience, model } = contextData;
  
  // Get agent instructions to check for custom questions that should always be asked
  const agentInstructions = await storage.getActiveAgentInstructions();
  const instructions = agentInstructions?.content || '';
  
  // Check if agent instructions contain "always ask" requirements
  const hasAlwaysAskRequirements = instructions.includes('always ask') || instructions.includes('Always ask');
  
  const confidence = imageAnalysis.confidence || 0.5;
  
  // If agent has "always ask" instructions, generate questions regardless of confidence
  if (hasAlwaysAskRequirements) {
    console.log(`Agent instructions require questions - generating questions (confidence: ${confidence})`);
  } else if (confidence > 0.75) {
    console.log(`High confidence (${confidence}) - skipping questions`);
    return [];
  } else {
    console.log(`Low confidence (${confidence}) - generating questions`);
  }
  
  const analysisPrompt = `
You are an AI wound care specialist following specific agent instructions. Based on the image analysis and agent guidelines, generate appropriate questions.

AGENT INSTRUCTIONS:
${instructions}

WOUND ANALYSIS RESULTS:
${JSON.stringify(imageAnalysis, null, 2)}

TARGET AUDIENCE: ${audience}

QUESTION GENERATION RULES:
1. Carefully read the AGENT INSTRUCTIONS above
2. Look for any "always ask" or "Always ask" requirements in the instructions
3. Generate questions based ONLY on what the Agent Instructions specify
4. If Agent Instructions require questions regardless of confidence, generate them
5. Generate questions WITHOUT pre-filled answers (leave answer field empty)
6. Only generate questions that the Agent Instructions explicitly require

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

QUESTION CATEGORIES (use appropriate category based on question content):
- "location" - Body location and wound site questions
- "patient_info" - Age, demographics, and patient details
- "symptoms" - Pain, swelling, and symptom-related questions
- "medical_history" - Pre-existing conditions and medical background
- "wound_assessment" - Wound characteristics and changes
- "other" - Any other questions specified by Agent Instructions

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