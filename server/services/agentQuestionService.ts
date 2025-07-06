import { callOpenAI } from './openai';
import { callGemini } from './gemini';
import { storage } from '../storage';
import { InsertAgentQuestion } from '@shared/schema';

async function generateFeedbackBasedQuestions(
  classification: any,
  userFeedback: string,
  audience: string,
  model: string
): Promise<string[]> {
  try {
    const prompt = `
Based on the wound assessment and user feedback, generate specific clarifying questions.

WOUND CLASSIFICATION:
- Type: ${classification.woundType}
- Location: ${classification.location}
- Stage: ${classification.stage}
- Size: ${classification.size}

USER FEEDBACK:
"${userFeedback}"

INSTRUCTIONS:
1. Analyze the user feedback for contradictions or corrections to the visual assessment
2. Generate 2-4 specific questions to clarify the feedback
3. Focus on resolving discrepancies between visual assessment and user input
4. If user mentions wrong body part/location, ask for correct location
5. If user mentions different wound type, ask for clarification
6. Keep questions simple and targeted to the audience: ${audience}

Return only the questions, one per line, without numbering.
`;

    // Get AI instructions for system prompt
    const agentInstructions = await storage.getActiveAgentInstructions();
    if (!agentInstructions?.systemPrompts) {
      throw new Error('AI Configuration not found. Please configure AI system prompts in Settings.');
    }

    let questions: string;
    if (model.startsWith('gemini-')) {
      const fullPrompt = `${agentInstructions.systemPrompts}\n\n${prompt}`;
      questions = await callGemini(model, fullPrompt);
    } else {
      const messages = [
        { role: "system", content: agentInstructions.systemPrompts },
        { role: "user", content: prompt }
      ];
      questions = await callOpenAI(model, messages);
    }
    
    return questions.split('\n')
      .map(q => q.trim())
      .filter(q => q.length > 0 && !q.match(/^\d+\./) && q.includes('?'))
      .slice(0, 4);
      
  } catch (error) {
    console.error('Error generating feedback-based questions:', error);
    return [];
  }
}

export async function analyzeAssessmentForQuestions(
  sessionIdOrClassification: string | any,
  contextDataOrPreviousQuestions?: any,
  userFeedback?: string,
  audienceParam?: string,
  modelParam?: string
): Promise<any[]> {
  // Support both original function signature and new feedback-based signature
  let sessionId: string;
  let contextData: any;
  
  if (typeof sessionIdOrClassification === 'string') {
    // Original function call
    sessionId = sessionIdOrClassification;
    contextData = contextDataOrPreviousQuestions;
  } else {
    // New feedback-based call
    const classification = sessionIdOrClassification;
    const previousQuestions = contextDataOrPreviousQuestions;
    
    // Handle feedback-based question generation
    if (userFeedback && userFeedback.trim() !== '') {
      return await generateFeedbackBasedQuestions(classification, userFeedback, audienceParam || 'patient', modelParam || 'gpt-4o');
    }
    
    return [];
  }
  const { imageAnalysis, audience, model, previousQuestions, round, instructions: providedInstructions } = contextData;
  
  // Get agent instructions to check for custom questions that should always be asked
  const agentInstructions = await storage.getActiveAgentInstructions();
  // Build complete instructions from structured fields if not provided
  let instructions = providedInstructions;
  if (!instructions && agentInstructions) {
    instructions = `
${agentInstructions.systemPrompts || ''}
${agentInstructions.carePlanStructure || ''}
${agentInstructions.specificWoundCare || ''}
${agentInstructions.questionsGuidelines || ''}
${agentInstructions.productRecommendations || ''}
`.trim();
  } else if (!instructions) {
    throw new Error('AI Configuration not found. Please configure AI instructions in Settings.');
  }
  
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
  
  // Handle follow-up questions differently than initial questions
  const isFollowUp = previousQuestions && previousQuestions.length > 0;
  const currentRound = round || 1;
  
  if (isFollowUp && currentRound > 1) {
    // For follow-up questions, be much more selective
    if (confidence > 0.80) {
      console.log(`Follow-up round ${currentRound}: High confidence (${confidence}) - skipping additional questions`);
      return [];
    } else {
      console.log(`Follow-up round ${currentRound}: Low confidence (${confidence}) - checking if more questions needed`);
    }
  } else {
    // Initial questions - check Agent Instructions requirements
    if (hasQuestionRequirements) {
      console.log(`Agent instructions require questions - generating questions (confidence: ${confidence})`);
    } else if (confidence > 0.80) {
      console.log(`High confidence (${confidence}) and no question requirements - skipping questions`);
      return [];
    } else {
      console.log(`Low confidence (${confidence}) - generating questions`);
    }
  }

  const analysisPrompt = `
You are an AI wound care specialist following specific agent instructions. ${isFollowUp ? 'This is a follow-up round of questions.' : 'This is the initial question generation.'}

AGENT INSTRUCTIONS:
${instructions}

WOUND ANALYSIS RESULTS:
${JSON.stringify(imageAnalysis, null, 2)}

CONFIDENCE ASSESSMENT:
Current confidence level: ${Math.round(confidence * 100)}%
${confidence < 0.50 ? 'Very low confidence - additional photos and detailed information needed' : 
  confidence < 0.80 ? 'Low confidence - more questions and possibly additional photos needed' : 
  'High confidence - minimal additional questions needed'}

${isFollowUp ? `PREVIOUS QUESTIONS AND ANSWERS (Round ${currentRound - 1}):
${JSON.stringify(previousQuestions, null, 2)}

FOLLOW-UP ASSESSMENT:
- Review the previous answers to see if they provide sufficient information
- Check if Agent Instructions require additional specific questions
- Only generate NEW questions if there are gaps that need clarification
- Maximum 3 rounds total, this is round ${currentRound}
` : ''}

TARGET AUDIENCE: ${audience}

QUESTION STRATEGY FRAMEWORK:

A) CONFIDENCE IMPROVEMENT QUESTIONS (when confidence < 80%):
   Focus on clarifying what was unclear in the image analysis:
   - Location specifics: "Where exactly on the body is this wound located?"
   - Medical history: "Do you have diabetes or circulation problems?"
   - Wound bed characteristics: "What color is the wound bed (red, yellow, black)?"
   - Wound edges: "Are the wound edges raised, flat, or undermined?"
   - Timeline: "How long have you had this wound?"
   - Origin: "How did this wound occur?"

B) CARE PLAN OPTIMIZATION QUESTIONS (when confidence ≥ 80%):
   Focus on treatment planning and symptom management:
   - Symptoms: "Do you experience pain, numbness, or swelling around the wound?"
   - Infection confirmation: "Is there increased warmth, red streaking, or foul odor?"
   - Drainage details: "What type and amount of drainage do you see?"
   - Current care: "What treatments have you tried so far?"
   - Progress tracking: "Have you noticed any improvements or worsening?"

C) MEDICAL REFERRAL QUESTIONS (when referral indicated):
   Focus on information doctors need:
   - Wound duration: "Exactly how long have you had this wound?"
   - Injury mechanism: "What caused this wound initially?"
   - Previous treatments: "What medical treatments have you received?"
   - Associated symptoms: "Any fever, increased pain, or other concerning symptoms?"
   - Medical conditions: "Any diabetes, circulation issues, or immune problems?"

PHOTO SUGGESTIONS (when confidence < 70%):
${contextData.imageCount > 1 ? 
  `Current images: ${contextData.imageCount} photos provided
- If wound edges still unclear: "Could you upload a clearer close-up photo of the wound edges?"
- If size uncertain: "Could you upload a photo with a reference object (coin/ruler) for accurate measurements?"
- If depth unclear: "Could you upload additional side-angle photos to show wound depth?"
- If lighting inconsistent: "Could you upload a photo with consistent, bright lighting?"` :
  `Single image provided - additional angles recommended:
- If wound edges unclear: "Could you upload a clearer photo of the wound edges?"
- If size uncertain: "Could you upload a photo with a reference object for size comparison?"
- If depth unclear: "Could you upload a photo from a different angle?"
- If lighting poor: "Could you upload a photo with better lighting?"`}

QUESTION SELECTION STRATEGY:
Current confidence: ${contextData.imageAnalysis.confidence}
- If confidence < 80%: Focus on Category A (confidence improvement) questions
- If confidence ≥ 80%: Focus on Category B (care plan optimization) questions  
- If medical referral suspected: Include Category C (doctor preparation) questions
- If confidence < 70%: Include photo suggestions

Generate 2-4 strategically selected questions based on:
1. What's unclear from the image analysis
2. Current confidence level
3. Information gaps that would most improve the assessment
4. Whether referral to medical professional is likely needed

${isFollowUp ? 'This is a follow-up round of questions. Only ask additional questions if the Agent Instructions require them or if confidence is still below 80%.' : 'Generate initial questions based strictly on what the Agent Instructions specify, plus photo suggestions if confidence is low.'}

RESPONSE FORMAT:
Return a JSON array of objects with this structure:
[
  {
    "id": "q1",
    "question": "Question as specified by Agent Instructions",
    "answer": "",
    "category": "category_name",
    "confidence": 0.0
  }
]

Use appropriate categories: location, patient_info, symptoms, medical_history, wound_assessment, photo_request, other
`;

  try {
    let response: string;
    
    if (model && model.startsWith('gemini')) {
      try {
        response = await callGemini(model, analysisPrompt);
      } catch (geminiError: any) {
        // Check if this is a quota error
        if (geminiError.message?.includes('quota') || geminiError.message?.includes('RESOURCE_EXHAUSTED')) {
          console.log('AgentQuestionService: Gemini service temporarily unavailable, automatically switching to GPT-4o');
          // Automatically switch to GPT-4o when Gemini service is unavailable
          const messages = [
            {
              role: "user",
              content: analysisPrompt
            }
          ];
          response = await callOpenAI('gpt-4o', messages);
        } else {
          // Re-throw non-quota errors
          throw geminiError;
        }
      }
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