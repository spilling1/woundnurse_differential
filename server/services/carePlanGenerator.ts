import { callOpenAI } from "./openai";
import { callGemini } from "./gemini";
import { getPromptTemplate } from "./promptTemplates";

export async function generateCarePlan(
  audience: string,
  classification: any, 
  contextData?: any,
  model: string = 'gpt-4o',
  imageData?: string,
  imageMimeType?: string
): Promise<string> {
  try {
    const prompt = await getPromptTemplate(audience, classification, contextData);
    
    let carePlan;
    
    if (model.startsWith('gemini-')) {
      const systemPrompt = "You are a medical AI assistant specializing in wound care. Generate comprehensive, evidence-based care plans tailored to the specified audience.";
      const fullPrompt = `${systemPrompt}\n\n${prompt}`;
      if (imageData) {
        carePlan = await callGemini(model, fullPrompt, imageData);
      } else {
        carePlan = await callGemini(model, fullPrompt);
      }
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
      
      if (imageData) {
        // Add image to the user message for vision models
        messages[1] = {
          role: "user",
          content: [
            {
              type: "text",
              text: prompt
            },
            {
              type: "image_url",
              image_url: {
                url: `data:${imageMimeType || 'image/jpeg'};base64,${imageData}`
              }
            }
          ]
        } as any;
      }
      
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
