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
1. First, check the AGENT INSTRUCTIONS for any "always ask" requirements:
   - Always ask about body location if not clear
   - Always ask about patient age unless irrelevant
   - Always ask about pain, swelling, and concerning symptoms
   - Always ask about pre-existing conditions
2. Generate questions for any "always ask" requirements that apply to this case
3. If confidence is low (<75%), also generate additional diagnostic clarification questions
4. Generate 2-4 targeted questions WITHOUT pre-filled answers

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