import { callOpenAI } from "./openai";
import { callGemini } from "./gemini";
import { getPromptTemplate } from "./promptTemplates";

export async function generateCarePlan(
  classification: any, 
  audience: string, 
  model: string,
  contextData?: any
): Promise<string> {
  try {
    const prompt = await getPromptTemplate(audience, classification, contextData);
    
    let carePlan;
    
    if (model.startsWith('gemini-')) {
      const systemPrompt = "You are a medical AI assistant specializing in wound care. Generate comprehensive, evidence-based care plans tailored to the specified audience.";
      const fullPrompt = `${systemPrompt}\n\n${prompt}`;
      carePlan = await callGemini(model, fullPrompt);
    } else {
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
      carePlan = await callOpenAI(model, messages);
    }
    
    // Add safety disclaimer
    const disclaimer = "**MEDICAL DISCLAIMER:** This is an AI-generated plan. Please consult a healthcare professional before following recommendations.";
    
    return `${disclaimer}\n\n${carePlan}`;
    
  } catch (error: any) {
    console.error('Care plan generation error:', error);
    throw new Error(`Care plan generation failed: ${error.message}`);
  }
}
