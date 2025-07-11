import { callOpenAI } from './openai';
import { callGemini } from './gemini';
import { storage } from '../storage';
import { InsertAgentQuestion } from '@shared/schema';

// Helper function to validate and correct question audience addressing
function validateAndCorrectAudience(questions: any[], audience: string): any[] {
  if (!questions || questions.length === 0) return questions;
  
  return questions.map(q => {
    let correctedQuestion = q.question;
    
    if (audience === 'medical' || audience === 'family') {
      // For medical professionals and family caregivers, use third person
      // Replace common second-person patterns with third-person equivalents
      correctedQuestion = correctedQuestion
        .replace(/\bDo you have\b/gi, 'Does the patient have')
        .replace(/\bHave you experienced\b/gi, 'Has the patient experienced')
        .replace(/\bHave you noticed\b/gi, audience === 'medical' ? 'Have you observed' : 'Have you noticed')
        .replace(/\bAre you experiencing\b/gi, 'Is the patient experiencing')
        .replace(/\bIs this wound causing you\b/gi, 'Is this wound causing the patient')
        .replace(/\bWhat treatments have you tried\b/gi, 
          audience === 'medical' ? 'What treatments has the patient received' : 'What treatments have you tried for the patient')
        .replace(/\byour wound\b/gi, "the patient's wound")
        .replace(/\byour family member\b/gi, "the patient")
        .replace(/\byour leg\b/gi, "the patient's leg")
        .replace(/\byour foot\b/gi, "the patient's foot")
        .replace(/\byour skin\b/gi, "the patient's skin")
        .replace(/\byour diabetes\b/gi, "the patient's diabetes")
        .replace(/\byour medical\b/gi, "the patient's medical")
        .replace(/\byour mobility\b/gi, "the patient's mobility")
        .replace(/\byour ability\b/gi, "the patient's ability");
    } else if (audience === 'patient') {
      // For patients, ensure second person is used
      correctedQuestion = correctedQuestion
        .replace(/\bDoes the patient have\b/gi, 'Do you have')
        .replace(/\bHas the patient experienced\b/gi, 'Have you experienced')
        .replace(/\bIs the patient experiencing\b/gi, 'Are you experiencing')
        .replace(/\bIs this wound causing the patient\b/gi, 'Is this wound causing you')
        .replace(/\bWhat treatments has the patient received\b/gi, 'What treatments have you tried')
        .replace(/\bthe patient's wound\b/gi, "your wound")
        .replace(/\bthe patient's leg\b/gi, "your leg")
        .replace(/\bthe patient's foot\b/gi, "your foot")
        .replace(/\bthe patient's skin\b/gi, "your skin")
        .replace(/\bthe patient's diabetes\b/gi, "your diabetes")
        .replace(/\bthe patient's medical\b/gi, "your medical")
        .replace(/\bthe patient's mobility\b/gi, "your mobility")
        .replace(/\bthe patient's ability\b/gi, "your ability");
    }
    
    return {
      ...q,
      question: correctedQuestion
    };
  });
}

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
  const { imageAnalysis, audience, model, previousQuestions, round, instructions: providedInstructions, bodyRegion } = contextData;
  
  // Log body region information if provided
  if (bodyRegion) {
    console.log('AgentQuestionService: Body region information received:', {
      id: bodyRegion.id,
      name: bodyRegion.name
    });
  } else {
    console.log('AgentQuestionService: No body region information provided');
  }
  
  // Get agent instructions to check for custom questions that should always be asked
  const agentInstructions = await storage.getActiveAgentInstructions();
  
  // Get wound-type-specific instructions if we have a classification
  let woundTypeInstructions = '';
  if (imageAnalysis.woundType) {
    try {
      console.log(`AgentQuestionService: Looking up wound type "${imageAnalysis.woundType}"`);
      const woundType = await storage.getWoundTypeByName(imageAnalysis.woundType);
      if (woundType && woundType.instructions) {
        woundTypeInstructions = `\n\nWOUND TYPE SPECIFIC INSTRUCTIONS FOR ${woundType.displayName.toUpperCase()}:\n${woundType.instructions}`;
        console.log(`AgentQuestionService: Found wound type instructions for ${woundType.displayName}`);
        console.log(`AgentQuestionService: Instructions contain "MUST ASK":`, woundType.instructions.includes('MUST ASK'));
      } else {
        console.log(`AgentQuestionService: No wound type found or no instructions for "${imageAnalysis.woundType}"`);
      }
    } catch (error) {
      console.error('Error getting wound type instructions:', error);
    }
  } else {
    console.log('AgentQuestionService: No wound type in imageAnalysis');
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
  
  // Add body region information if provided
  if (bodyRegion) {
    const bodyRegionContext = `\n\nBODY REGION CONTEXT:\nThe wound is located on: ${bodyRegion.name}\nThis anatomical location may provide important context for:\n- Wound type classification (e.g., diabetic ulcers commonly on feet, pressure ulcers on bony prominences)\n- Risk factors and contributing factors\n- Healing considerations and treatment approach\n- Differential diagnosis considerations\n- Specific questions related to anatomical location\n\nConsider this location information when generating appropriate questions.`;
    instructions += bodyRegionContext;
    console.log('AgentQuestionService: Added body region context to AI instructions:', bodyRegionContext);
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
  
  // Check if we have differential diagnosis information
  const hasDifferentialDiagnosis = differentialDiagnosis && differentialDiagnosis.possibleTypes && differentialDiagnosis.possibleTypes.length > 1;
  
  // Generate default differential diagnosis questions if not provided by AI
  let defaultDifferentialQuestions = [];
  if (hasDifferentialDiagnosis && aiSuggestedQuestions.length === 0) {
    const possibleTypes = differentialDiagnosis.possibleTypes.map(t => t.woundType.toLowerCase());
    
    // Add diabetes questions for diabetic/pressure ulcer differentiation
    if (possibleTypes.some(t => t.includes('diabetic')) && possibleTypes.some(t => t.includes('pressure'))) {
      defaultDifferentialQuestions.push("Does the patient have diabetes or a history of high blood sugar?");
      defaultDifferentialQuestions.push("Is the patient able to feel sensation in the area around the wound?");
      defaultDifferentialQuestions.push("How much time does the patient spend sitting or lying down each day?");
    }
    
    // Add circulation questions for venous/arterial differentiation
    if (possibleTypes.some(t => t.includes('venous')) && possibleTypes.some(t => t.includes('arterial'))) {
      defaultDifferentialQuestions.push("Does the patient have swelling in their legs or ankles?");
      defaultDifferentialQuestions.push("Are the patient's feet often cold or have poor circulation?");
    }
    
    // Add general origin questions for traumatic wounds
    if (possibleTypes.some(t => t.includes('traumatic'))) {
      defaultDifferentialQuestions.push("How did this wound first occur?");
      defaultDifferentialQuestions.push("Was there a specific injury or accident that caused this wound?");
    }
    
    console.log('AgentQuestionService: Generated default differential questions:', defaultDifferentialQuestions);
  }
  
  // Wound type detection is now handled by database-driven instructions
  
  const confidence = imageAnalysis.confidence || 0.5;
  
  // Handle follow-up questions differently than initial questions
  const isFollowUp = previousQuestions && previousQuestions.length > 0;
  const currentRound = round || 1;
  
  // All question requirements are now database-driven
  
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
    // Initial questions - follow database instructions for question requirements
    console.log(`GENERATE QUESTIONS PER DATABASE INSTRUCTIONS (confidence: ${confidence})`);
    if (hasWoundTypeRequirements) {
      console.log(`WOUND TYPE REQUIREMENTS DETECTED - MUST ask specific questions regardless of confidence`);
    }
    if (hasQuestionRequirements) {
      console.log(`Agent instructions contain question requirements from database`);
    }
  }

  // Extract differential diagnosis information and questions from AI analysis
  const differentialDiagnosis = imageAnalysis.differentialDiagnosis;
  const aiSuggestedQuestions = differentialDiagnosis?.questionsToAsk || [];
  
  console.log('AgentQuestionService: Differential diagnosis info:', differentialDiagnosis);
  console.log('AgentQuestionService: AI suggested questions:', aiSuggestedQuestions);

  const analysisPrompt = `
You are an AI wound care specialist following specific agent instructions. ${isFollowUp ? 'This is a follow-up round of questions.' : 'This is the initial question generation.'}

AGENT INSTRUCTIONS:
${instructions}

WOUND ANALYSIS RESULTS:
${JSON.stringify(imageAnalysis, null, 2)}

DIFFERENTIAL DIAGNOSIS INFORMATION:
${differentialDiagnosis ? `The AI has identified multiple possible wound types:
${differentialDiagnosis.possibleTypes.map(type => `- ${type.woundType}: ${Math.round(type.confidence * 100)}% confidence - ${type.reasoning}`).join('\n')}

AI SUGGESTED QUESTIONS TO DIFFERENTIATE:
${aiSuggestedQuestions.map(q => `- ${q}`).join('\n')}

PRIORITY: Ask questions that help distinguish between these possibilities. Focus on:
1. Medical history questions (diabetes, circulation, mobility)
2. Symptom questions (pain, numbness, drainage)
3. Wound origin questions (how it started, what caused it)
4. Location-specific questions (pressure, activity level)` : 'No differential diagnosis information available.'}

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

CRITICAL AUDIENCE-SPECIFIC COMMUNICATION RULES:
${audience === 'medical' ? `
🩺 MEDICAL PROFESSIONAL AUDIENCE:
- Address the medical professional as a healthcare provider caring for a patient
- Use THIRD PERSON when asking about the patient (e.g., "Has the patient experienced fever?" NOT "Have you experienced fever?")
- Use professional medical terminology
- Ask about patient history, symptoms, and treatments in third person
- Example: "Does the patient have a history of diabetes?"
- Example: "Has the patient experienced any systemic signs of infection?"
- Example: "What treatments has the patient received for this wound?"
` : audience === 'family' ? `
👨‍👩‍👧‍👦 FAMILY CAREGIVER AUDIENCE:
- Address them as caregivers helping a patient/family member
- Use THIRD PERSON when asking about the patient (e.g., "Has your family member shown signs of infection?" NOT "Have you shown signs of infection?")
- Use clear, non-technical language
- Ask about the patient's condition from caregiver's perspective
- Example: "Does the patient have diabetes or other medical conditions?"
- Example: "Has the patient mentioned any pain or discomfort?"
- Example: "Have you noticed any changes in the patient's wound?"
` : `
🧑‍🦽 PATIENT AUDIENCE:
- Address the patient directly in SECOND PERSON
- Use "you" and "your" when asking about their condition
- Use clear, empathetic language
- Example: "Do you have diabetes or other medical conditions?"
- Example: "Have you experienced any pain or discomfort?"
- Example: "Have you noticed any changes in your wound?"
`}

CRITICAL WOUND TYPE REQUIREMENTS:
${hasWoundTypeRequirements ? `
🚨 MANDATORY WOUND TYPE REQUIREMENTS DETECTED 🚨

CRITICAL REQUIREMENT: EXTRACT EVERY "MUST ASK" LINE FROM DATABASE INSTRUCTIONS

Step 1: LINE-BY-LINE SCAN
Scan the Agent Instructions above and find these EXACT lines:
- "MUST ASK - ORIGIN OF THE WOUND" 
- "MUST ASK - TETANUS STATUS"
- "MUST ASK - CONTAMINATION RISK" 
- "MUST ASK - ""WHY IS YOUR HAIR RED?"""

Step 2: MANDATORY QUESTION GENERATION
For EACH "MUST ASK" line found, you MUST generate ONE question:

1. ORIGIN OF THE WOUND → Ask how/when the injury occurred
2. TETANUS STATUS → Ask about tetanus vaccination status  
3. CONTAMINATION RISK → Ask about environment where injury occurred and what contaminated it
4. "WHY IS YOUR HAIR RED?" → Include this EXACT question as written in database

🚨 VERIFICATION CHECKLIST 🚨
Count the "MUST ASK" lines in the instructions. You MUST generate exactly that many questions.
For traumatic wounds, you should find 4 "MUST ASK" requirements.
If you generate fewer than 4 questions, you have FAILED to follow database instructions.

CRITICAL: Generate questions for ALL "MUST ASK" requirements. Do NOT paraphrase, summarize, or skip any requirements.
IMPORTANT: Filter out any test questions or non-medical questions (e.g., questions about hair color, personal appearance, etc.).
` : ''}

QUESTION STRATEGY FRAMEWORK:

GENERATE MAXIMUM 5 QUESTIONS PER PAGE - Focus on HIGH and MEDIUM priority questions only

A) HIGH PRIORITY QUESTIONS (Essential for accurate assessment):
${audience === 'medical' ? `
   - Medical history: "Does the patient have diabetes?"
   - Wound origin: "How did the patient's wound occur?"
   - Timeline: "How long has the patient had this wound?"
   - Location specifics: "Where exactly on the patient's body is this wound located?"
   - Infection signs: "Do you observe increased redness, warmth, or pus around the patient's wound?"
` : audience === 'family' ? `
   - Medical history: "Does the patient have diabetes?"
   - Wound origin: "How did the patient's wound occur?"
   - Timeline: "How long has the patient had this wound?"
   - Location specifics: "Where exactly on the patient's body is this wound located?"
   - Infection signs: "Have you noticed increased redness, warmth, or pus around the patient's wound?"
` : `
   - Medical history: "Do you have diabetes?"
   - Wound origin: "How did this wound occur?"
   - Timeline: "How long have you had this wound?"
   - Location specifics: "Where exactly on your body is this wound located?"
   - Infection signs: "Do you see increased redness, warmth, or pus around your wound?"
`}

B) MEDIUM PRIORITY QUESTIONS (Improve care plan quality):
${audience === 'medical' ? `
   - Current symptoms: "Does the patient experience pain or numbness around the wound?"
   - Drainage: "What type and amount of drainage do you observe from the patient's wound?"
   - Current treatments: "What treatments has the patient received so far?"
   - Progress: "Has the patient's wound changed in size or appearance recently?"
   - Mobility impact: "Is this wound affecting the patient's ability to walk or move?"
` : audience === 'family' ? `
   - Current symptoms: "Has the patient mentioned pain or numbness around the wound?"
   - Drainage: "What type and amount of drainage have you noticed from the patient's wound?"
   - Current treatments: "What treatments have you tried for the patient so far?"
   - Progress: "Has the patient's wound changed in size or appearance recently?"
   - Mobility impact: "Is this wound affecting the patient's ability to walk or move?"
` : `
   - Current symptoms: "Do you experience pain or numbness around the wound?"
   - Drainage: "What type and amount of drainage do you see from your wound?"
   - Current treatments: "What treatments have you tried so far?"
   - Progress: "Has your wound changed in size or appearance recently?"
   - Mobility impact: "Is this wound affecting your ability to walk or move?"
`}

C) LOW PRIORITY QUESTIONS (Skip unless critical):
   - Wound bed color details
   - Specific wound measurements
   - Detailed drainage characteristics
   - Minor symptom variations

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
- Extract ALL requirements directly from the database instructions provided above

` : `
Standard confidence-based strategy:
- If confidence < 80%: Focus on Category A (confidence improvement) questions
- If confidence ≥ 80%: Focus on Category B (care plan optimization) questions  
- If medical referral suspected: Include Category C (doctor preparation) questions
- If confidence < 70%: Include photo suggestions
`}

🚨 FOLLOW DATABASE INSTRUCTIONS: Follow all requirements specified in Agent Instructions 🚨

REQUIREMENTS FROM DATABASE:
- Follow ALL requirements specified in the Agent Instructions (configured in Settings)
- Follow wound-type specific "MUST ASK" or "Clarifying Questions" from Wound Type Instructions
- Follow any other requirements specified in the database-stored Agent Instructions

Generate questions based on:
1. ${hasWoundTypeRequirements ? 'FIRST: Required wound-type specific questions from Agent Instructions (these are MANDATORY)' : 'Wound-type specific requirements if any'}
2. ALL requirements specified in Agent Instructions database
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
1. MAXIMUM 5 QUESTIONS PER PAGE - This is a hard limit
2. Each question must be a SINGLE, STANDALONE question
3. Do NOT combine multiple questions into one
4. Do NOT add advice, instructions, or commentary within questions
5. Do NOT use "Please avoid..." or similar advisory language in questions
6. Each question should only ask ONE thing
7. Questions must end with a question mark
8. Keep questions simple and direct
9. PRIORITIZE HIGH-IMPACT QUESTIONS: Focus on diabetes history, wound origin, infection signs, timeline, and current treatments only

🚨 MANDATORY AUDIENCE ENFORCEMENT 🚨
${audience === 'medical' ? `
FOR MEDICAL PROFESSIONALS - USE THIRD PERSON ONLY:
- NEVER use "you" or "your" when asking about the patient
- ALWAYS use "the patient" or "patient" when referring to the person with the wound
- ALWAYS use "you" when asking about the medical professional's observations
- CORRECT: "Does the patient have diabetes?"
- INCORRECT: "Do you have diabetes?"
- CORRECT: "Have you observed any changes in the patient's wound?"
- INCORRECT: "Have you noticed any changes in your wound?"
` : audience === 'family' ? `
FOR FAMILY CAREGIVERS - USE THIRD PERSON ONLY:
- NEVER use "you" or "your" when asking about the patient
- ALWAYS use "the patient" or "patient" when referring to the person with the wound
- ALWAYS use "you" when asking about the caregiver's observations
- CORRECT: "Does the patient have diabetes?"
- INCORRECT: "Do you have diabetes?"
- CORRECT: "Have you noticed any changes in the patient's wound?"
- INCORRECT: "Have you noticed any changes in your wound?"
` : `
FOR PATIENTS - USE SECOND PERSON ONLY:
- ALWAYS use "you" and "your" when asking about their condition
- CORRECT: "Do you have diabetes?"
- CORRECT: "Have you noticed any changes in your wound?"
`}

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
${audience === 'medical' ? `
✓ "Does the patient have diabetes?"
✓ "Have you observed any changes in redness or swelling around the patient's wound?"
✓ "What treatments has the patient received so far?"
` : audience === 'family' ? `
✓ "Does the patient have diabetes?"
✓ "Have you noticed any changes in redness or swelling around the patient's wound?"
✓ "What treatments have you tried for the patient so far?"
` : `
✓ "Do you have diabetes?"
✓ "Have you noticed any changes in redness or swelling around your wound?"
✓ "What treatments have you tried so far?"
`}

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
        
        // Filter out test questions and non-medical questions
        const isTestQuestion = 
          cleanQuestion.toLowerCase().includes('hair') ||
          cleanQuestion.toLowerCase().includes('color') ||
          cleanQuestion.toLowerCase().includes('eye') ||
          cleanQuestion.toLowerCase().includes('appearance') ||
          cleanQuestion.toLowerCase().includes('test') ||
          cleanQuestion.toLowerCase().includes('red') && cleanQuestion.toLowerCase().includes('hair');
        
        const isValidMedicalQuestion = isValidQuestion && !isTestQuestion;
        
        return {
          ...q,
          id: q.id || `q${index + 1}`,
          question: isValidMedicalQuestion ? cleanQuestion.trim() : `Is this wound causing you any pain or discomfort?`,
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
            if (similarity > 0.6) { // Lowered threshold for better duplicate detection
              console.log(`Removing similar question: "${newQ.question}" (similarity: ${similarity})`);
              return true;
            }
            return false;
          });
          
          return !isDuplicate;
        });
        
        console.log(`Filtered ${cleanedQuestions.length - filteredQuestions.length} duplicate questions`);
      }
      
      // Remove duplicates within the current set of questions
      const uniqueQuestions = [];
      const seenQuestions = new Set();
      
      for (const question of filteredQuestions) {
        const questionLower = question.question.toLowerCase().trim();
        let isDuplicate = false;
        
        // Check for exact matches
        if (seenQuestions.has(questionLower)) {
          console.log(`Removing internal duplicate: "${question.question}"`);
          continue;
        }
        
        // Check for semantic duplicates within current set
        for (const existingQuestion of uniqueQuestions) {
          const similarity = calculateQuestionSimilarity(existingQuestion.question, questionLower);
          if (similarity > 0.6) {
            console.log(`Removing internal similar question: "${question.question}" (similarity: ${similarity})`);
            isDuplicate = true;
            break;
          }
        }
        
        if (!isDuplicate) {
          uniqueQuestions.push(question);
          seenQuestions.add(questionLower);
        }
      }
      
      // Validate and correct audience addressing before finalizing
      const audienceValidatedQuestions = validateAndCorrectAudience(uniqueQuestions, audience);
      
      // ENFORCE MAXIMUM 5 QUESTIONS PER PAGE
      const MAX_QUESTIONS_PER_PAGE = 5;
      const finalQuestions = audienceValidatedQuestions.slice(0, MAX_QUESTIONS_PER_PAGE);
      
      console.log(`Question generation summary: Generated ${cleanedQuestions.length} → Filtered ${filteredQuestions.length} → Unique ${uniqueQuestions.length} → Audience-validated ${audienceValidatedQuestions.length} → Final ${finalQuestions.length} (max ${MAX_QUESTIONS_PER_PAGE})`);
      
      filteredQuestions = finalQuestions;
      
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
      
      // All question requirements are now database-driven through AI instructions
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
    
    // Log the error for analysis
    if (sessionId) {
      try {
        await storage.createAiInteraction({
          caseId: sessionId,
          stepType: 'question_generation',
          modelUsed: model || 'gpt-4o',
          promptSent: 'AI question generation failed completely',
          responseReceived: '',
          parsedResult: null,
          confidenceScore: Math.round(confidence * 100),
          errorOccurred: true,
          errorMessage: `Complete AI failure: ${error.message || error}`,
        });
      } catch (logError) {
        console.error('Error logging complete AI failure:', logError);
      }
    }
    
    // Return error instead of fallback questions
    throw new Error('The image could not be processed, please try again.');
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