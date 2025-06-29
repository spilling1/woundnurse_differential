import { storage } from "../storage";

export async function getPromptTemplate(audience: string, classification: any, contextData?: any): Promise<string> {
  // Get agent instructions from database
  const agentInstructions = await storage.getActiveAgentInstructions();
  const instructions = agentInstructions?.content || `Default wound care guidelines: Always prioritize patient safety and recommend consulting healthcare professionals.`;
  let baseInfo = `
Current Wound Assessment:
- Type: ${classification.woundType}
- Stage: ${classification.stage}
- Size: ${classification.size}
- Wound Bed: ${classification.woundBed}
- Exudate: ${classification.exudate}
- Location: ${classification.location}
- Infection Signs: ${classification.infectionSigns.join(', ') || 'None observed'}
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

  const systemPrompt = `You are an AI wound care specialist. Follow these instructions:

${instructions}

${baseInfo}`;

  switch (audience) {
    case 'family':
      const familyFollowUpInstructions = contextData?.isFollowUp ? `

IMPORTANT FOR FOLLOW-UP ASSESSMENT:
- Start with a "Progress Summary" section comparing current status to previous assessments
- Highlight what has improved, stayed the same, or worsened since the last assessment
- Reference previous care plan recommendations and note which should continue, be modified, or stopped
- Address the treatment response noted in the progress report
- Provide updated recommendations based on current wound status
- Acknowledge the caregiver's efforts and provide encouragement where appropriate` : '';

      return `${systemPrompt}

Generate a comprehensive wound care plan for FAMILY CAREGIVERS with the following requirements:

1. Use clear, non-medical language that is easy to understand
2. Provide step-by-step instructions for wound care
3. Include practical tips for home care
4. Specify when to seek professional help
5. List warning signs to watch for
6. Include frequency of care and monitoring${familyFollowUpInstructions}

Structure the response with clear sections:${contextData?.isFollowUp ? '\n- Progress Summary (compare to previous assessments)' : ''}
- Cleaning Instructions
- Dressing Recommendations (use generic names, not brands)
- Frequency of Care
- Warning Signs
- When to Contact Healthcare Provider
- Additional Tips for Caregivers`;

    case 'patient':
      const patientFollowUpInstructions = contextData?.isFollowUp ? `

IMPORTANT FOR FOLLOW-UP ASSESSMENT:
- Begin with a "Your Progress" section celebrating improvements and addressing any concerns
- Compare current wound status to previous assessments in encouraging, understandable terms
- Reference previous self-care recommendations and update them based on current progress
- Address the patient's treatment response and progress notes
- Provide motivation and realistic expectations for continued healing
- Adjust care routine based on current wound status and patient feedback` : '';

      return `${systemPrompt}

Generate a comprehensive wound care plan for PATIENTS with the following requirements:

1. Use empowering, clear language that builds confidence
2. Focus on self-care and independence
3. Provide educational information about healing process
4. Include lifestyle recommendations
5. Emphasize importance of following medical advice
6. Address common concerns and fears${patientFollowUpInstructions}

Structure the response with clear sections:${contextData?.isFollowUp ? '\n- Your Progress (celebrating improvements and addressing concerns)' : ''}
- Understanding Your Wound
- Self-Care Instructions
- Daily Care Routine
- Signs of Healing vs. Concern
- Lifestyle Factors for Healing
- When to Seek Help
- Encouragement and Expectations`;

    case 'medical':
      const medicalFollowUpInstructions = contextData?.isFollowUp ? `

IMPORTANT FOR FOLLOW-UP ASSESSMENT:
- Lead with "Clinical Progress Assessment" section providing objective comparison to baseline and previous assessments
- Document wound progression using standardized terminology and measurements
- Reference previous treatment protocols and provide clinical rationale for continuing, modifying, or discontinuing interventions
- Analyze treatment response documented in progress notes with clinical interpretation
- Recommend evidence-based protocol adjustments based on current wound status
- Include specific monitoring parameters and reassessment timeline` : '';

      return `${systemPrompt}

Generate a comprehensive wound care plan for MEDICAL PROFESSIONALS with the following requirements:

1. Use clinical terminology and evidence-based protocols
2. Include assessment parameters and monitoring criteria
3. Provide treatment rationale and alternatives
4. Include documentation requirements${medicalFollowUpInstructions}
5. Reference clinical guidelines where appropriate
6. Address complications and management strategies

Structure the response with clear sections:${contextData?.isFollowUp ? '\n- Clinical Progress Assessment (objective comparison to previous assessments)' : ''}
- Clinical Assessment Summary
- Evidence-Based Treatment Protocol
- Dressing Selection Rationale
- Monitoring Parameters
- Expected Outcomes Timeline
- Complication Management
- Documentation Requirements
- Follow-up Recommendations`;

    default:
      throw new Error(`Invalid audience type: ${audience}`);
  }
}
