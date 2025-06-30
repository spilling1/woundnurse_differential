import { callOpenAI } from './openai';
import { callGemini } from './gemini';
import { storage } from '../storage';
import { InsertAgentQuestion } from '@shared/schema';

export async function analyzeAssessmentForQuestions(
  sessionId: string,
  contextData: any
): Promise<any[]> {
  const { imageAnalysis, audience, model, previousQuestions, round, instructions: providedInstructions } = contextData;
  
  // Get agent instructions to check for custom questions that should always be asked
  const agentInstructions = await storage.getActiveAgentInstructions();
  const instructions = providedInstructions || agentInstructions?.content || '';
  
  // Check if agent instructions contain question requirements
  const instructionsLower = instructions.toLowerCase();
  const hasQuestionRequirements = 
    instructions.includes('always ask') || 
    instructions.includes('Always ask') ||
    instructions.includes('follow-up questions') ||
    instructions.includes('Follow-up questions') ||
    instructions.includes('Ask at least') ||
    instructions.includes('ask at least') ||
    instructionsLower.includes('follow-up') ||
    (instructionsLower.includes('ask') && instructionsLower.includes('question'));
  
  const confidence = imageAnalysis.confidence || 0.5;
  
  // If agent has question requirements, generate questions regardless of confidence
  if (hasQuestionRequirements) {
    console.log(`Agent instructions require questions - generating questions (confidence: ${confidence})`);
  } else if (confidence > 0.75) {
    console.log(`High confidence (${confidence}) and no question requirements - skipping questions`);
    return [];
  } else {
    console.log(`Low confidence (${confidence}) - generating questions`);
  }
  
  // Handle follow-up questions differently than initial questions
  const isFollowUp = previousQuestions && previousQuestions.length > 0;
  const currentRound = round || 1;

  const analysisPrompt = `
You are an AI wound care specialist following specific agent instructions. ${isFollowUp ? 'This is a follow-up round of questions.' : 'This is the initial question generation.'}

AGENT INSTRUCTIONS:
${instructions}

WOUND ANALYSIS RESULTS:
${JSON.stringify(imageAnalysis, null, 2)}

${isFollowUp ? `PREVIOUS QUESTIONS AND ANSWERS (Round ${currentRound - 1}):
${JSON.stringify(previousQuestions, null, 2)}

FOLLOW-UP ASSESSMENT:
- Review the previous answers to see if they provide sufficient information
- Check if Agent Instructions require additional specific questions
- Only generate NEW questions if there are gaps that need clarification
- Maximum 3 rounds total, this is round ${currentRound}
` : ''}

TARGET AUDIENCE: ${audience}

QUESTION GENERATION RULES:
${isFollowUp ? `
1. Review previous answers to determine if Agent Instructions are satisfied
2. Only ask follow-up questions if Agent Instructions require more specific information
3. Do NOT repeat questions that have already been answered
4. Check if diagnostic confidence can be improved with additional clarification
5. Generate 0-2 targeted follow-up questions, or empty array if no more questions needed
` : `
1. Carefully read the AGENT INSTRUCTIONS above
2. Look for any "always ask" or "Always ask" requirements in the instructions  
3. Generate questions based ONLY on what the Agent Instructions specify
4. If Agent Instructions require questions regardless of confidence, generate them
5. Generate 2-4 initial questions based on Agent Instructions requirements
`}
6. Generate questions WITHOUT pre-filled answers (leave answer field empty)
7. Only generate questions that the Agent Instructions explicitly require

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