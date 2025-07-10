import { callOpenAI } from './openai';
import { callGemini } from './gemini';
import { storage } from '../storage';
import { InsertAgentQuestion } from '@shared/schema';

// Helper function to calculate similarity between two questions
function calculateQuestionSimilarity(question1: string, question2: string): number {
  // Normalize questions by removing common words and focusing on key terms
  const normalize = (q: string) => q
    .toLowerCase()
    .replace(/[^\w\s]/g, '')
    .replace(/\b(do|you|have|are|is|any|the|a|an|this|that|your|what|where|when|how|why)\b/g, '')
    .trim()
    .split(/\s+/)
    .filter(word => word.length > 2);
  
  const words1 = normalize(question1);
  const words2 = normalize(question2);
  
  if (words1.length === 0 || words2.length === 0) return 0;
  
  // Calculate Jaccard similarity (intersection over union)
  const set1 = new Set(words1);
  const set2 = new Set(words2);
  const intersection = new Set([...set1].filter(x => set2.has(x)));
  const union = new Set([...set1, ...set2]);
  
  return intersection.size / union.size;
}

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
  
  // Get wound-type-specific instructions if we have a classification
  let woundTypeInstructions = '';
  if (imageAnalysis.woundType) {
    try {
      const woundType = await storage.getWoundTypeByName(imageAnalysis.woundType);
      if (woundType && woundType.instructions) {
        woundTypeInstructions = `\n\nWOUND TYPE SPECIFIC INSTRUCTIONS FOR ${woundType.display_name.toUpperCase()}:\n${woundType.instructions}`;
      }
    } catch (error) {
      console.error('Error getting wound type instructions:', error);
    }
  }
  
  // Build complete instructions from structured fields if not provided
  let instructions = providedInstructions;
  if (!instructions && agentInstructions) {
    instructions = `
${agentInstructions.systemPrompts || ''}
${agentInstructions.carePlanStructure || ''}
${agentInstructions.specificWoundCare || ''}
${agentInstructions.questionsGuidelines || ''}
${agentInstructions.productRecommendations || ''}
${woundTypeInstructions}
`.trim();
  } else if (!instructions) {
    instructions = woundTypeInstructions || '';
    if (!instructions) {
      throw new Error('AI Configuration not found. Please configure AI instructions in Settings.');
    }
  } else {
    // Add wound type instructions to provided instructions
    instructions = `${instructions}${woundTypeInstructions}`;
  }
  
  // Check if agent instructions contain question requirements
  const instructionsLower = instructions.toLowerCase();
  const hasQuestionRequirements = 
    instructions.includes('always ask') || 
    instructions.includes('Always ask') ||
    instructions.includes('MUST ASK') ||
    instructions.includes('Must ask') ||
    instructions.includes('follow-up questions') ||
    instructions.includes('Follow-up questions') ||
    instructions.includes('Ask at least') ||
    instructions.includes('ask at least') ||
    instructionsLower.includes('follow-up') ||
    (instructionsLower.includes('ask') && instructionsLower.includes('question'));
    
  // Specifically check for wound-type specific requirements
  const hasWoundTypeRequirements = 
    instructions.includes('MUST ASK') ||
    instructions.includes('Clarifying Questions:') ||
    instructionsLower.includes('origin of the wound') ||
    instructionsLower.includes('exactly how and when did it happen');
  
  const confidence = imageAnalysis.confidence || 0.5;
  
  // Handle follow-up questions differently than initial questions
  const isFollowUp = previousQuestions && previousQuestions.length > 0;
  const currentRound = round || 1;
  
  if (isFollowUp && currentRound > 1) {
    // For follow-up questions, check if the answers provided require reassessment
    const hasSignificantAnswers = previousQuestions.some((q: any) => 
      q.answer && q.answer.trim().length > 0 && 
      (q.answer.toLowerCase().includes('diabetes') || 
       q.answer.toLowerCase().includes('suicide') || 
       q.answer.toLowerCase().includes('amputation') ||
       q.answer.toLowerCase().includes('numbness') ||
       q.answer.toLowerCase().includes('infection') ||
       q.answer.toLowerCase().includes('pain') ||
       q.answer.toLowerCase().includes('yes') ||
       q.answer.toLowerCase().includes('no'))
    );
    
    if (hasSignificantAnswers) {
      console.log(`Follow-up round ${currentRound}: Processing user answers for reassessment - confidence: ${confidence}`);
    } else if (confidence >= 1.0 && currentRound > 2) {
      console.log(`Follow-up round ${currentRound}: 100% confidence and round 3+ - skipping additional questions`);
      return [];
    } else {
      console.log(`Follow-up round ${currentRound}: Low confidence (${confidence}) - checking if more questions needed`);
    }
  } else {
    // Initial questions - check Agent Instructions requirements
    if (hasQuestionRequirements || hasWoundTypeRequirements) {
      console.log(`Agent instructions require questions - generating questions (confidence: ${confidence})`);
      if (hasWoundTypeRequirements) {
        console.log(`Wound type specific requirements detected - must ask origin questions`);
      }
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

CRITICAL FOLLOW-UP ASSESSMENT STEPS:
1. REASSESS WOUND CLASSIFICATION: Carefully review user answers for information that contradicts or significantly changes the initial wound assessment
2. IDENTIFY CONTRADICTIONS: Check if user answers contradict the visual assessment (e.g., "I don't have diabetes" vs "diabetic ulcer" classification)
3. DETECT CRITICAL INFORMATION: Look for serious medical concerns like diabetes, suicide risk, infection signs, numbness, or mobility issues
4. DETERMINE NEED FOR RECLASSIFICATION: If answers provide significant new information, consider whether wound type or urgency level should change
5. GENERATE TARGETED QUESTIONS: Only ask NEW questions if critical information is missing or contradictory
6. MAXIMUM 3 ROUNDS: This is round ${currentRound} - be increasingly selective

REASSESSMENT TRIGGERS:
- User confirms or denies diabetes (impacts diabetic ulcer classification)
- User mentions suicide, depression, or amputation fears (requires mental health protocols)
- User reports numbness, inability to walk (suggests neurological involvement)
- User confirms/denies infection signs (impacts urgency and treatment)
- User provides contradictory location or wound origin information
` : ''}

TARGET AUDIENCE: ${audience}

CRITICAL WOUND TYPE REQUIREMENTS:
${hasWoundTypeRequirements ? `
⚠️ MANDATORY WOUND TYPE REQUIREMENTS DETECTED ⚠️
The Agent Instructions contain SPECIFIC requirements for this wound type that MUST be followed:
- Look for "MUST ASK" requirements in the instructions
- Look for "Clarifying Questions" sections in the instructions  
- Follow ALL wound-type-specific question requirements regardless of confidence level
- These requirements override general confidence-based question strategies

For traumatic wounds specifically:
- MUST ask about origin/mechanism of injury
- MUST ask "How did this wound occur?" or similar
- MUST ask about timing "When did it happen?"
- MUST ask about foreign material or contamination
` : ''}

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

${hasWoundTypeRequirements ? `
🚨 PRIORITY REQUIREMENT: WOUND TYPE SPECIFIC QUESTIONS MUST BE ASKED FIRST 🚨
1. FIRST: Check Agent Instructions for wound-type specific "MUST ASK" or "Clarifying Questions" requirements
2. Generate ALL required wound-type questions regardless of confidence level
3. THEN consider general assessment questions if needed

For this specific wound type, look for and include:
- Questions marked as "MUST ASK" in the Agent Instructions
- Questions listed under "Clarifying Questions:" sections
- Origin/mechanism questions for traumatic wounds
- Timing and contamination questions

` : `
Standard confidence-based strategy:
- If confidence < 80%: Focus on Category A (confidence improvement) questions
- If confidence ≥ 80%: Focus on Category B (care plan optimization) questions  
- If medical referral suspected: Include Category C (doctor preparation) questions
- If confidence < 70%: Include photo suggestions
`}

Generate 2-4 strategically selected questions based on:
1. ${hasWoundTypeRequirements ? 'FIRST: Required wound-type specific questions from Agent Instructions' : 'What\'s unclear from the image analysis'}
2. Current confidence level
3. Information gaps that would most improve the assessment
4. Whether referral to medical professional is likely needed

${isFollowUp ? `
CRITICAL FOLLOW-UP REQUIREMENT:
Before generating any new questions, you MUST first reassess the wound classification based on the user's answers. If the user's answers contradict the initial visual assessment or provide significant new information, you should:

1. Note the contradiction or new information
2. Suggest a revised wound classification if appropriate
3. Address any unusual, contradictory, or concerning responses
4. Only then determine if additional questions are needed

For example:
- If image suggested "diabetic ulcer" but user says "I don't have diabetes" → reassess as pressure ulcer or venous ulcer
- If user mentions "suicidal thoughts" or "amputation fears" → flag for mental health protocols
- If user reports "numbness" or "can't walk" → consider neurological involvement
- If user gives contradictory explanations (e.g., "hot metal" vs typical neuropathic ulcer patterns) → question the explanation and note the contradiction
- If user mentions dangerous treatments (e.g., "soaking in whiskey") → STRONGLY address safety concerns, explain why harmful, and provide proper wound care guidance
- If user claims wound is from trauma but image shows characteristics of systemic disease → address the discrepancy
- If user denies diabetes but wound shows classic diabetic ulcer characteristics → explain the medical evidence and recommend screening

MANDATORY REASSESSMENT RESPONSE FORMAT:
First provide a reassessment section like this:
REASSESSMENT: [Explain how the user's answers impact the wound classification and care plan - this is critical. Address any contradictory, unusual, or concerning responses directly.]

Then provide questions (if needed) in JSON format.

This is a follow-up round of questions. Only ask additional questions if the Agent Instructions require them or if confidence is still below 80%.
` : 'Generate initial questions based strictly on what the Agent Instructions specify, plus photo suggestions if confidence is low.'}

CRITICAL RESPONSE FORMAT REQUIREMENTS:
${isFollowUp ? `
You MUST respond with a reassessment section followed by a JSON array:

REASSESSMENT: [Your analysis of how the user's answers impact the wound classification and care plan]

Then provide the JSON array format below.
` : 'You MUST respond with ONLY a valid JSON array. Do NOT include any other text, explanations, or formatting.'}

CRITICAL QUESTION FORMATTING RULES:
1. Each question must be a SINGLE, STANDALONE question
2. Do NOT combine multiple questions into one
3. Do NOT add advice, instructions, or commentary within questions
4. Do NOT use "Please avoid..." or similar advisory language in questions
5. Each question should only ask ONE thing
6. Questions must end with a question mark
7. Keep questions simple and direct

${isFollowUp ? `
CRITICAL DUPLICATE PREVENTION:
The following questions have ALREADY been asked in previous rounds:
${previousQuestions.map((q: any) => `- ${q.question}`).join('\n')}

YOU MUST NOT ask any of these questions again, even with different wording. Generate only NEW questions that have NOT been asked before.
` : ''}

REQUIRED JSON FORMAT:
[
  {
    "id": "q1",
    "question": "Single, standalone question only?",
    "answer": "",
    "category": "category_name",
    "confidence": 0.0
  }
]

VALID CATEGORIES: location, patient_info, symptoms, medical_history, wound_assessment, photo_request, other

EXAMPLE OF CORRECT QUESTIONS:
✓ "Do you have diabetes?"
✓ "Have you noticed any changes in redness or swelling?"
✓ "What treatments have you tried so far?"

EXAMPLE OF INCORRECT QUESTIONS:
✗ "Please avoid using urine or other non-medical substances on the wound. Have you noticed any changes in the redness or swelling?"
✗ "Do you have diabetes? This is important for wound healing."
✗ "What treatments have you tried? Please only use medical treatments."

IMPORTANT: Your response must be valid JSON that can be parsed directly. ${isFollowUp ? 'Include the reassessment section before the JSON array.' : 'Do not include any text before or after the JSON array.'}
`;

  try {
    let response: string;
    
    if (model && model.startsWith('gemini')) {
      try {
        // Add system instruction for Gemini to ensure proper response format
        const systemInstruction = isFollowUp ? 
          "You are a medical AI assistant that MUST respond with a reassessment section followed by valid JSON arrays. In your reassessment, address ANY contradictory, unusual, or medically concerning responses directly. Format: REASSESSMENT: [your analysis] then JSON array. Your response must be parseable JSON." :
          "You are a medical AI assistant that MUST respond with valid JSON arrays only. Never include explanatory text, conversation, or any content outside the JSON format. Your response must be parseable JSON.";
        const fullPrompt = `${systemInstruction}\n\n${analysisPrompt}`;
        response = await callGemini(model, fullPrompt);
      } catch (geminiError: any) {
        // Check if this is a quota error
        if (geminiError.message?.includes('quota') || geminiError.message?.includes('RESOURCE_EXHAUSTED')) {
          console.log('AgentQuestionService: Gemini service temporarily unavailable, automatically switching to GPT-4o');
          // Automatically switch to GPT-4o when Gemini service is unavailable
          const messages = [
            {
              role: "system",
              content: isFollowUp ? 
                "You are a medical AI assistant that MUST respond with a reassessment section followed by valid JSON arrays. In your reassessment, address ANY contradictory, unusual, or medically concerning responses directly. Format: REASSESSMENT: [your analysis] then JSON array. Your response must be parseable JSON." :
                "You are a medical AI assistant that MUST respond with valid JSON arrays only. Never include explanatory text, conversation, or any content outside the JSON format. Your response must be parseable JSON."
            },
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
          role: "system",
          content: isFollowUp ? 
            "You are a medical AI assistant that MUST respond with a reassessment section followed by valid JSON arrays. In your reassessment, address ANY contradictory, unusual, or medically concerning responses directly. Format: REASSESSMENT: [your analysis] then JSON array. Your response must be parseable JSON." :
            "You are a medical AI assistant that MUST respond with valid JSON arrays only. Never include explanatory text, conversation, or any content outside the JSON format. Your response must be parseable JSON."
        },
        {
          role: "user",
          content: analysisPrompt
        }
      ];
      response = await callOpenAI(model || 'gpt-4o', messages);
    }

    // Clean the response to remove any markdown formatting and extract JSON
    let cleanedResponse = response
      .replace(/```json/g, '')
      .replace(/```/g, '')
      .trim();
      
    // For follow-up responses, extract reassessment information
    let reassessmentText = '';
    if (isFollowUp) {
      const reassessmentMatch = cleanedResponse.match(/REASSESSMENT:\s*(.*?)(?=\[)/s);
      if (reassessmentMatch) {
        reassessmentText = reassessmentMatch[1].trim();
        console.log('Reassessment extracted:', reassessmentText);
      }
    }
    
    // Try to extract JSON array from the response if it contains other text
    const jsonMatch = cleanedResponse.match(/\[[\s\S]*\]/);
    if (jsonMatch) {
      cleanedResponse = jsonMatch[0];
    }
    
    // Remove any leading/trailing text that's not part of the JSON
    cleanedResponse = cleanedResponse.replace(/^[^[]*/, '').replace(/[^\]]*$/, '');
    
    // Ensure we have a valid JSON array structure
    if (!cleanedResponse.startsWith('[') || !cleanedResponse.endsWith(']')) {
      console.error('AI response does not contain valid JSON array:', response);
      // Return empty array instead of throwing error to prevent analysis failure
      return [];
    }
      
    try {
      const questions = JSON.parse(cleanedResponse);
      
      // Clean and validate questions to prevent combining or inappropriate content
      const cleanedQuestions = Array.isArray(questions) ? questions.map((q, index) => {
        let cleanQuestion = q.question;
        
        // Remove any advisory language or combined statements
        cleanQuestion = cleanQuestion
          // Remove "Please avoid..." type statements
          .replace(/^Please avoid[^.]*\.\s*/i, '')
          .replace(/^Avoid[^.]*\.\s*/i, '')
          // Remove instructional prefixes
          .replace(/^Note:[^.]*\.\s*/i, '')
          .replace(/^Important:[^.]*\.\s*/i, '')
          .replace(/^Remember:[^.]*\.\s*/i, '')
          // Split on sentence boundaries and take only the question part
          .split(/\.\s+(?=[A-Z])/)
          .find(part => part.includes('?')) || cleanQuestion;
        
        // Ensure it ends with a question mark
        if (!cleanQuestion.endsWith('?')) {
          cleanQuestion += '?';
        }
        
        // Validate that it's actually a question
        const isValidQuestion = cleanQuestion.includes('?') && 
          (cleanQuestion.toLowerCase().startsWith('do ') ||
           cleanQuestion.toLowerCase().startsWith('have ') ||
           cleanQuestion.toLowerCase().startsWith('what ') ||
           cleanQuestion.toLowerCase().startsWith('where ') ||
           cleanQuestion.toLowerCase().startsWith('when ') ||
           cleanQuestion.toLowerCase().startsWith('how ') ||
           cleanQuestion.toLowerCase().startsWith('why ') ||
           cleanQuestion.toLowerCase().startsWith('is ') ||
           cleanQuestion.toLowerCase().startsWith('are ') ||
           cleanQuestion.toLowerCase().startsWith('can ') ||
           cleanQuestion.toLowerCase().startsWith('could ') ||
           cleanQuestion.toLowerCase().startsWith('would '));
        
        return {
          ...q,
          id: q.id || `q${index + 1}`,
          question: isValidQuestion ? cleanQuestion.trim() : `Is this wound causing you any pain or discomfort?`,
          answer: q.answer || '',
          category: q.category || 'symptoms',
          confidence: q.confidence || 0
        };
      }).filter(q => q.question && q.question.length > 5) : [];
      
      // Remove duplicate questions by checking against previous questions
      let filteredQuestions = cleanedQuestions;
      if (isFollowUp && previousQuestions && previousQuestions.length > 0) {
        const previousQuestionTexts = previousQuestions.map((pq: any) => pq.question.toLowerCase().trim());
        
        filteredQuestions = cleanedQuestions.filter(newQ => {
          const newQuestionLower = newQ.question.toLowerCase().trim();
          
          // Check for exact matches
          if (previousQuestionTexts.includes(newQuestionLower)) {
            console.log(`Removing exact duplicate question: "${newQ.question}"`);
            return false;
          }
          
          // Check for semantic duplicates (similar meaning)
          const isDuplicate = previousQuestionTexts.some(prevQ => {
            const similarity = calculateQuestionSimilarity(prevQ, newQuestionLower);
            if (similarity > 0.7) {
              console.log(`Removing similar question: "${newQ.question}" (similarity: ${similarity})`);
              return true;
            }
            return false;
          });
          
          return !isDuplicate;
        });
        
        console.log(`Filtered ${cleanedQuestions.length - filteredQuestions.length} duplicate questions`);
      }
      
      // Log the question generation AI interaction
      if (sessionId) {
        try {
          await storage.createAiInteraction({
            caseId: sessionId,
            stepType: 'question_generation',
            modelUsed: model || 'gpt-4o',
            promptSent: analysisPrompt,
            responseReceived: reassessmentText ? `REASSESSMENT: ${reassessmentText}\n\n${cleanedResponse}` : cleanedResponse,
            parsedResult: { questions: cleanedQuestions, reassessment: reassessmentText },
            confidenceScore: Math.round(confidence * 100),
            errorOccurred: false,
          });
        } catch (logError) {
          console.error('Error logging question generation AI interaction:', logError);
        }
      }
      
      return filteredQuestions;
    } catch (parseError: any) {
      console.error('Failed to parse AI questions response:', cleanedResponse);
      
      // Log the error
      if (sessionId) {
        try {
          await storage.createAiInteraction({
            caseId: sessionId,
            stepType: 'question_generation',
            modelUsed: model || 'gpt-4o',
            promptSent: analysisPrompt,
            responseReceived: cleanedResponse,
            parsedResult: null,
            confidenceScore: Math.round(confidence * 100),
            errorOccurred: true,
            errorMessage: `JSON parsing failed: ${parseError.message}`,
          });
        } catch (logError) {
          console.error('Error logging question generation error:', logError);
        }
      }
      
      throw new Error(`Invalid JSON response from AI model: ${parseError.message}`);
    }
    
  } catch (error) {
    console.error('Error generating AI questions:', error);
    // Re-throw the error so it can be handled by the calling code
    throw new Error(`Failed to generate AI questions: ${error.message || error}`);
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