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
    
    baseInfo += `

**IMPORTANT DIAGNOSTIC INSTRUCTION:** 
Category A answers MUST take precedence over visual image analysis when there's a contradiction. 
For example, if image suggests "diabetic ulcer" but patient clearly states "I do not have diabetes", 
then reclassify as a different wound type (pressure ulcer, venous ulcer, etc.) based on location and features.
Category B answers should enhance treatment recommendations and symptom management.
Category C answers should influence urgency level and medical referral recommendations.
Always explain your reasoning when patient answers contradict initial visual assessment.
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

Generate a well-formatted care plan that follows the structure guidelines above.`;

  return fullPrompt;
}
