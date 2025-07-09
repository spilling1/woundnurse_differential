import { storage } from "../storage";

export async function getPromptTemplate(audience: string, classification: any, contextData?: any, relevantProducts?: any[]): Promise<string> {
  // Get agent instructions from database
  const agentInstructions = await storage.getActiveAgentInstructions();
  
  // Ensure AI instructions are configured
  if (!agentInstructions) {
    throw new Error('AI Configuration not found. Please configure AI instructions in Settings before generating care plans.');
  }

  // Build the complete instructions from the structured fields only
  const instructions = `
${agentInstructions.systemPrompts || ''}

CARE PLAN STRUCTURE GUIDELINES:
${agentInstructions.carePlanStructure || ''}

SPECIFIC WOUND CARE INSTRUCTIONS:
${agentInstructions.specificWoundCare || ''}

QUESTIONS GUIDELINES:
${agentInstructions.questionsGuidelines || ''}

PRODUCT RECOMMENDATIONS GUIDELINES:
${agentInstructions.productRecommendations || ''}

IMPORTANT: For "Items to Purchase" section, you MUST provide specific, working product recommendations with proper Amazon search links:
- Use format: [Product Name](https://www.amazon.com/s?k=search+terms+here)
- Replace spaces with + in search terms
- Be specific with wound care products based on actual assessment
- Include multiple options for each category (dressings, cleansers, etc.)
- Example: [Foam Dressing for Wounds](https://www.amazon.com/s?k=foam+dressing+wounds+adhesive)
- Example: [Saline Wound Cleanser](https://www.amazon.com/s?k=saline+wound+cleanser+sterile)
`.trim();
  

  let baseInfo = `
Current Wound Assessment:
- Type: ${classification.woundType}
- Stage: ${classification.stage}
- Size: ${classification.size}
- Wound Bed: ${classification.woundBed}
- Exudate: ${classification.exudate}
- Location: ${classification.location}
- Infection Signs: ${classification.infectionSigns?.join(', ') || 'None observed'}
- Additional Observations: ${classification.additionalObservations}

Patient Context:
${contextData?.woundOrigin ? `- How/when wound occurred: ${contextData.woundOrigin}` : ''}
${contextData?.medicalHistory ? `- Medical history: ${contextData.medicalHistory}` : ''}
${contextData?.woundChanges ? `- Recent changes noted: ${contextData.woundChanges}` : ''}
${contextData?.currentCare ? `- Current care being provided: ${contextData.currentCare}` : ''}
${contextData?.woundPain ? `- Pain description: ${contextData.woundPain}` : ''}
${contextData?.supportAtHome ? `- Support at home: ${contextData.supportAtHome}` : ''}
${contextData?.mobilityStatus ? `- Mobility status: ${contextData.mobilityStatus}` : ''}
${contextData?.nutritionStatus ? `- Nutrition status: ${contextData.nutritionStatus}` : ''}
`;

  // Process AI Questions and Answers by Category
  console.log('DEBUG: promptTemplates - contextData.aiQuestions:', JSON.stringify(contextData?.aiQuestions, null, 2));
  if (contextData?.aiQuestions && Array.isArray(contextData.aiQuestions)) {
    const categorizedQuestions: {
      confidenceImprovement: any[];
      carePlanOptimization: any[];
      medicalReferral: any[];
    } = {
      confidenceImprovement: [],
      carePlanOptimization: [],
      medicalReferral: []
    };
    
    // Categorize questions based on content
    contextData.aiQuestions.forEach((qa: any) => {
      if (qa.answer && qa.answer.trim() !== '') {
        const question = qa.question.toLowerCase();
        
        // Category A: Confidence Improvement Questions
        if (question.includes('diabetes') || question.includes('where') || 
            question.includes('location') || question.includes('how long') ||
            question.includes('wound bed') || question.includes('color') ||
            question.includes('how did') || question.includes('occur')) {
          categorizedQuestions.confidenceImprovement.push(qa);
        }
        // Category B: Care Plan Optimization Questions
        else if (question.includes('pain') || question.includes('drainage') ||
                 question.includes('treatment') || question.includes('infection') ||
                 question.includes('numbness') || question.includes('swelling') ||
                 question.includes('improvement') || question.includes('tried')) {
          categorizedQuestions.carePlanOptimization.push(qa);
        }
        // Category C: Medical Referral Questions
        else if (question.includes('fever') || question.includes('medical') ||
                 question.includes('doctor') || question.includes('symptoms') ||
                 question.includes('circulation') || question.includes('immune')) {
          categorizedQuestions.medicalReferral.push(qa);
        }
        // Default to confidence improvement
        else {
          categorizedQuestions.confidenceImprovement.push(qa);
        }
      }
    });

    baseInfo += `

CRITICAL: Patient Answers by Category (MUST OVERRIDE VISUAL CLASSIFICATION IF ANSWERS CONTRADICT):

A) DIAGNOSTIC CONFIDENCE ANSWERS (Impact wound classification):
`;
    categorizedQuestions.confidenceImprovement.forEach((qa: any) => {
      baseInfo += `- Q: ${qa.question}\n- A: ${qa.answer}\n`;
    });

    if (categorizedQuestions.carePlanOptimization.length > 0) {
      baseInfo += `
B) CARE PLAN OPTIMIZATION ANSWERS (Impact treatment recommendations):
`;
      categorizedQuestions.carePlanOptimization.forEach((qa: any) => {
        baseInfo += `- Q: ${qa.question}\n- A: ${qa.answer}\n`;
      });
    }

    if (categorizedQuestions.medicalReferral.length > 0) {
      baseInfo += `
C) MEDICAL REFERRAL PREPARATION ANSWERS (Impact urgency/referral need):
`;
      categorizedQuestions.medicalReferral.forEach((qa: any) => {
        baseInfo += `- Q: ${qa.question}\n- A: ${qa.answer}\n`;
      });
    }
    
    // Check for mental health concerns and dangerous treatments in question answers
    let mentalHealthConcerns = '';
    let dangerousTreatments = '';
    contextData.aiQuestions.forEach((qa: any) => {
      if (qa.answer && qa.answer.trim() !== '') {
        const answer = qa.answer.toLowerCase();
        
        // Check for depression/suicide indicators
        if (answer.includes('suicide') || answer.includes('kill myself') || 
            answer.includes('end it all') || answer.includes('not worth living') ||
            answer.includes('want to die') || answer.includes('better off dead')) {
          mentalHealthConcerns += `CRITICAL MENTAL HEALTH ALERT: Suicide risk detected in answer: "${qa.answer}"\n`;
        } else if (answer.includes('depression') || answer.includes('depressed') ||
                   answer.includes('hopeless') || answer.includes('overwhelmed') ||
                   answer.includes('giving up') || answer.includes('can\'t cope')) {
          mentalHealthConcerns += `MENTAL HEALTH CONCERN: Depression indicators detected in answer: "${qa.answer}"\n`;
        }
        
        // Check for dangerous treatments
        if (answer.includes('whiskey') || answer.includes('alcohol') || 
            answer.includes('bleach') || answer.includes('peroxide') ||
            answer.includes('hot water') || answer.includes('ice') ||
            answer.includes('salt water') || answer.includes('vinegar') ||
            answer.includes('baking soda') || answer.includes('essential oil')) {
          dangerousTreatments += `DANGEROUS TREATMENT ALERT: Harmful treatment mentioned in answer: "${qa.answer}"\n`;
        }
      }
    });
    
    if (mentalHealthConcerns) {
      baseInfo += `

**CRITICAL MENTAL HEALTH SAFETY ALERT:**
${mentalHealthConcerns}
**MANDATORY ACTIONS FOR CARE PLAN:**
- Include suicide/mental health resources in URGENT MEDICAL ATTENTION section
- Recommend immediate professional mental health support
- For suicide risk: Include National Suicide Prevention Lifeline: 988 or 1-800-273-8255 in critical section
- For depression: Recommend contacting doctor or therapist
- Take all mental health concerns seriously regardless of wound severity
- QUOTE THE PATIENT'S SPECIFIC WORDS in "Your Specific Concerns Addressed" section
- Address their mental health crisis with empathy and urgency

`;
    }

    if (dangerousTreatments) {
      baseInfo += `

**DANGEROUS TREATMENT SAFETY ALERT:**
${dangerousTreatments}
**MANDATORY ACTIONS FOR CARE PLAN:**
- Address each dangerous treatment mentioned directly in "Your Specific Concerns Addressed" section
- QUOTE THE PATIENT'S SPECIFIC WORDS about treatments they mentioned
- Explain why these treatments are harmful (e.g., "soaking in whiskey can damage tissue and delay healing")
- Provide safe alternatives for wound care
- Include strong warnings in the care plan about avoiding these treatments
- Educate patient about proper wound care practices
- Include this in URGENT MEDICAL ATTENTION section if treatment is very dangerous

`;
    }

    baseInfo += `

**IMPORTANT DIAGNOSTIC INSTRUCTION:** 
Category A answers MUST take precedence over visual image analysis when there's a contradiction. 
For example, if image suggests "diabetic ulcer" but patient clearly states "I do not have diabetes", 
then reclassify as a different wound type (pressure ulcer, venous ulcer, etc.) based on location and features.
Category B answers should enhance treatment recommendations and symptom management.
Category C answers should influence urgency level and medical referral recommendations.
Always explain your reasoning when patient answers contradict initial visual assessment.

**CONTRADICTORY RESPONSE HANDLING:**
When patient answers contradict medical evidence or seem unusual, you MUST address them directly while explaining your medical reasoning:
- If wound patterns suggest neuropathic/diabetic ulcer but patient claims "hot metal" injury → respectfully disagree and explain why wound characteristics (bilateral location, punched-out appearance, lack of thermal injury signs) suggest neuropathic rather than thermal injury
- If patient mentions dangerous treatments (e.g., "soaking in whiskey") → STRONGLY advise against this practice, explain why alcohol can damage tissue and delay healing, and provide proper wound care alternatives
- If patient denies diabetes but shows classic diabetic ulcer signs → explain the medical evidence and recommend diabetes screening
- If patient provides inconsistent explanations → clarify the discrepancy and provide educational explanation
- Always acknowledge their response first, then explain your professional assessment with clear medical reasoning
- Use phrases like "I understand you mentioned X, however the wound characteristics suggest Y because..."
- Be respectful but firm when addressing dangerous practices - patient safety is paramount

**MANDATORY QUESTION ANSWER INTEGRATION:**
YOU MUST ADDRESS EACH QUESTION ANSWER SPECIFICALLY AND DIRECTLY. This is not optional.

For each answer provided by the user, you must:
1. Quote or reference the specific answer they gave
2. Explain how it impacts your assessment
3. Address any contradictions or concerns
4. Provide medical reasoning for your conclusions

Examples of REQUIRED integration:
- If user says "I don't think so?" about diabetes → "You mentioned you don't think you have diabetes, however the wound characteristics strongly suggest diabetic neuropathic ulcers..."
- If user describes impact on life → "You mentioned you can't walk and this is ruining your life as a dancer. I understand this is devastating..."
- If user mentions timeline → "You said these wounds have been present for about 2 months, which is concerning for..."
- If user mentions whiskey soaking → "You mentioned soaking your feet in whiskey every night. This is extremely dangerous and must be stopped immediately..."
- If user mentions suicide → "You mentioned you might commit suicide. This is a mental health emergency that requires immediate professional help..."

**MANDATORY SECTIONS TO INCLUDE:**
- Create a "Your Specific Concerns Addressed" section that directly responds to each answer
- For dangerous treatments or mental health concerns, include in urgent medical attention section
- For unrelated comments (like political questions), acknowledge but redirect: "You asked about political matters, but let's focus on your urgent medical needs..."
- Include phrase "I have taken into account your specific answers" in the care plan
- Never ignore or skip addressing any answer, even if it seems unrelated

**CRITICAL SAFETY REQUIREMENT:**
If mental health alerts or dangerous treatments are detected, they MUST be prominently featured in the care plan with specific quotes from patient answers.
`;
  }

  // Add user feedback from preliminary plan if provided
  if (contextData?.userFeedback && contextData.userFeedback.trim() !== '') {
    baseInfo += `

USER FEEDBACK ON PRELIMINARY PLAN:
The user provided the following feedback/corrections on the preliminary assessment:
"${contextData.userFeedback}"

IMPORTANT: Please incorporate this feedback into the final care plan. Address any corrections, concerns, or additional context provided by the user. This feedback should take priority over preliminary assessments when there are contradictions.
`;
  }

  // Add preliminary plan context if provided
  if (contextData?.preliminaryPlan) {
    baseInfo += `

PRELIMINARY PLAN CONTEXT:
This is a final care plan generation. The preliminary assessment showed:
- Confidence Level: ${Math.round((contextData.preliminaryPlan.confidence || 0.8) * 100)}%
- Assessment: ${contextData.preliminaryPlan.assessment || 'Not provided'}

Build upon this preliminary assessment while incorporating any user feedback provided above.
`;
  }

  // Add follow-up specific context if this is a follow-up assessment
  if (contextData?.isFollowUp && contextData?.previousAssessments) {
    baseInfo += `
FOLLOW-UP ASSESSMENT CONTEXT:
This is a follow-up assessment. Please reference the previous care plans and note changes in wound status.

Progress Notes: ${contextData.currentAssessment?.progressNotes || 'Not provided'}
Treatment Response: ${contextData.currentAssessment?.treatmentResponse || 'Not provided'}

Previous Assessment History:
`;

    // Add details about previous assessments
    contextData.previousAssessments.forEach((prev: any, index: number) => {
      baseInfo += `
Assessment ${index + 1} (${new Date(prev.createdAt).toLocaleDateString()}):
- Classification: ${prev.classification?.woundType} - ${prev.classification?.stage}
- Size: ${prev.classification?.size}
- Exudate: ${prev.classification?.exudate}
- Version: ${prev.versionNumber}
${prev.progressNotes ? `- Progress Notes: ${prev.progressNotes}` : ''}
${prev.treatmentResponse ? `- Treatment Response: ${prev.treatmentResponse}` : ''}
`;
    });

    if (contextData.feedbackHistory && contextData.feedbackHistory.length > 0) {
      baseInfo += `
Previous Feedback from Care Team:
`;
      contextData.feedbackHistory.forEach((feedback: any, index: number) => {
        baseInfo += `- ${feedback.feedbackType}: ${feedback.comments} (${new Date(feedback.createdAt).toLocaleDateString()})\n`;
      });
    }
  }

  // Build the complete prompt using only AI Configuration data
  const fullPrompt = `${instructions}

${baseInfo}

TARGET AUDIENCE: ${audience.toUpperCase()}
${contextData?.isFollowUp ? `
FOLLOW-UP ASSESSMENT: This is a follow-up assessment. Please compare current status to previous assessments and provide updated recommendations based on wound progression.` : ''}

**CRITICAL FORMATTING REQUIREMENTS - MUST FOLLOW:**
- Use proper markdown formatting throughout your response
- Start each section with bold headers: **SECTION NAME**
- Use bullet points (-) for lists and recommendations
- Use numbered lists (1., 2., 3.) for sequential care steps
- Separate sections with blank lines for readability
- Keep paragraphs concise and well-structured
- Format links properly as [Product Name](URL)
- Use proper spacing and indentation for sub-items

**ITEMS TO PURCHASE REQUIREMENTS:**
You MUST include a specific "ITEMS TO PURCHASE" section with products from our database:
${relevantProducts && relevantProducts.length > 0 ? `
**RECOMMENDED PRODUCTS (from our database):**
${relevantProducts.map(product => `
* [${product.name}](${product.amazonUrl}) - ${product.description}
  - Category: ${product.category}
  - Suitable for: ${product.woundTypes.join(', ')}
  - Target audience: ${product.audiences.join(', ')}
`).join('')}
` : `
**GENERAL PRODUCT RECOMMENDATIONS:**
- Working Amazon search links using format: https://www.amazon.com/s?k=product+search+terms
- At least 3-4 specific products relevant to wound type: ${classification.woundType}
- Examples:
  * [Hydrocolloid Dressings](https://www.amazon.com/s?k=hydrocolloid+dressings+wound+care)
  * [Medical Tape](https://www.amazon.com/s?k=medical+tape+hypoallergenic)
  * [Wound Cleansing Solution](https://www.amazon.com/s?k=wound+cleansing+solution+sterile)
  * [Antibacterial Ointment](https://www.amazon.com/s?k=antibacterial+ointment+wounds)
`}
IMPORTANT: Use the exact product names and Amazon URLs provided above. These are verified products from our database.

**QUESTION ANSWER HANDLING:**
- Always address medically relevant question answers directly within the appropriate sections of the care plan
- Integrate patient answers naturally into wound assessment, treatment recommendations, and care instructions
- For unrelated comments or non-medical questions, create a separate section at the bottom called "Additional Questions Addressed"
- Always include the phrase "I have taken into account your specific answers" somewhere in the care plan
- Take any mental health concerns (depression, suicide) extremely seriously - include appropriate resources and urgent referrals

**MENTAL HEALTH SAFETY PROTOCOL:**
- If patient mentions suicide, depression, or mental health struggles: Include National Suicide Prevention Lifeline: 988 or 1-800-273-8255
- For overt suicide references: Place hotline number in critical/urgent section with red styling
- For depression indicators: Recommend contacting doctor or therapist immediately
- Never minimize mental health concerns - treat them as seriously as physical wound care

Generate a well-formatted care plan that follows the structure guidelines above.`;

  return fullPrompt;
}
