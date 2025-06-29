import { callOpenAI } from "./openai";
import { getPromptTemplate } from "./promptTemplates";

export async function generateCarePlan(
  classification: any, 
  audience: string, 
  model: string
): Promise<string> {
  try {
    const prompt = getPromptTemplate(audience, classification);
    
    const messages = [
      {
        role: "system",
        content: "You are a medical AI assistant specializing in wound care. Generate comprehensive, evidence-based care plans tailored to the specified audience."
      },
      {
        role: "user",
        content: prompt
      }
    ];

    const carePlan = await callOpenAI(model, messages);
    
    // Add safety disclaimer
    const disclaimer = "**MEDICAL DISCLAIMER:** This is an AI-generated plan. Please consult a healthcare professional before following recommendations.";
    
    return `${disclaimer}\n\n${carePlan}`;
    
  } catch (error: any) {
    console.error('Care plan generation error:', error);
    throw new Error(`Care plan generation failed: ${error.message}`);
  }
}
