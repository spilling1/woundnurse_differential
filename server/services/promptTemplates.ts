import { storage } from "../storage";

export async function getPromptTemplate(audience: string, classification: any, contextData?: any): Promise<string> {
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

PRODUCT RECOMMENDATIONS:
${agentInstructions.productRecommendations || ''}
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

  // Process AI Questions and Answers
  if (contextData?.aiQuestions && Array.isArray(contextData.aiQuestions)) {
    baseInfo += `

CRITICAL: AI Follow-up Questions and Patient Answers (MUST OVERRIDE VISUAL CLASSIFICATION IF ANSWERS CONTRADICT):
`;
    contextData.aiQuestions.forEach((qa: any) => {
      if (qa.answer && qa.answer.trim() !== '') {
        baseInfo += `- Q: ${qa.question}\n- A: ${qa.answer}\n`;
      }
    });
    
    baseInfo += `

**IMPORTANT DIAGNOSTIC INSTRUCTION:** 
The patient answers above MUST take precedence over visual image analysis when there's a contradiction. 
For example, if image suggests "diabetic ulcer" but patient clearly states "I do not have diabetes", 
then reclassify as a different wound type (pressure ulcer, venous ulcer, etc.) based on location and features.
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
FOLLOW-UP ASSESSMENT: This is a follow-up assessment. Please compare current status to previous assessments and provide updated recommendations based on wound progression.` : ''}`;

  return fullPrompt;
}
