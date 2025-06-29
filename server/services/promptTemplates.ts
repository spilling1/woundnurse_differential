import { storage } from "../storage";

export async function getPromptTemplate(audience: string, classification: any, contextData?: any): Promise<string> {
  // Get agent instructions from database
  const agentInstructions = await storage.getActiveAgentInstructions();
  const instructions = agentInstructions?.content || `Default wound care guidelines: Always prioritize patient safety and recommend consulting healthcare professionals.`;
  const baseInfo = `
Wound Assessment:
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

  switch (audience) {
    case 'family':
      return `${baseInfo}

Generate a comprehensive wound care plan for FAMILY CAREGIVERS with the following requirements:

1. Use clear, non-medical language that is easy to understand
2. Provide step-by-step instructions for wound care
3. Include practical tips for home care
4. Specify when to seek professional help
5. List warning signs to watch for
6. Include frequency of care and monitoring

Structure the response with clear sections:
- Cleaning Instructions
- Dressing Recommendations (use generic names, not brands)
- Frequency of Care
- Warning Signs
- When to Contact Healthcare Provider
- Additional Tips for Caregivers`;

    case 'patient':
      return `${baseInfo}

Generate a comprehensive wound care plan for PATIENTS with the following requirements:

1. Use empowering, clear language that builds confidence
2. Focus on self-care and independence
3. Provide educational information about healing process
4. Include lifestyle recommendations
5. Emphasize importance of following medical advice
6. Address common concerns and fears

Structure the response with clear sections:
- Understanding Your Wound
- Self-Care Instructions
- Daily Care Routine
- Signs of Healing vs. Concern
- Lifestyle Factors for Healing
- When to Seek Help
- Encouragement and Expectations`;

    case 'medical':
      return `${baseInfo}

Generate a comprehensive wound care plan for MEDICAL PROFESSIONALS with the following requirements:

1. Use clinical terminology and evidence-based protocols
2. Include assessment parameters and monitoring criteria
3. Provide treatment rationale and alternatives
4. Include documentation requirements
5. Reference clinical guidelines where appropriate
6. Address complications and management strategies

Structure the response with clear sections:
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
