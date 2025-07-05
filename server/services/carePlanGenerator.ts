import { callOpenAI } from "./openai";
import { callGemini } from "./gemini";
import { getPromptTemplate } from "./promptTemplates";
import { storage } from "../storage";

export async function generateCarePlan(
  audience: string,
  classification: any, 
  contextData?: any,
  model: string = 'gpt-4o',
  imageData?: string,
  imageMimeType?: string,
  detectionInfo?: any
): Promise<string> {
  try {
    // Get AI instructions to use proper system prompts
    const agentInstructions = await storage.getActiveAgentInstructions();
    const systemPrompt = agentInstructions?.systemPrompts || 
      "You are a medical AI assistant specializing in wound care. Generate comprehensive, evidence-based care plans tailored to the specified audience.";
    
    const prompt = await getPromptTemplate(audience, classification, contextData);
    
    let carePlan;
    
    if (model.startsWith('gemini-')) {
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
          content: systemPrompt
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
    
    // Add detection system information if available
    let detectionSystemInfo = "";
    if (detectionInfo) {
      const detectionMethod = detectionInfo.model || 'Enhanced Fallback';
      const processingTime = detectionInfo.processingTime || 'N/A';
      const hasDetections = classification.detection;
      
      detectionSystemInfo = `\n\n---\n\n**DETECTION SYSTEM ANALYSIS:**\n
**Method Used:** ${detectionMethod}
**Processing Time:** ${processingTime}ms
**System Status:** ${detectionMethod === 'yolo9' ? 'YOLO9 Active' : detectionMethod === 'enhanced-fallback' ? 'Enhanced Fallback Mode' : 'Cloud Vision Active'}
${hasDetections ? `
**Detection Results:**
- Confidence: ${Math.round((hasDetections.confidence || 0) * 100)}%
- Wound Measurements:
  - Length: ${hasDetections.measurements?.lengthMm || 'N/A'}mm
  - Width: ${hasDetections.measurements?.widthMm || 'N/A'}mm  
  - Area: ${hasDetections.measurements?.areaMm2 || 'N/A'}mm²
- Scale Calibrated: ${hasDetections.scaleCalibrated ? 'Yes' : 'No'}
- Precise Measurements: ${classification.preciseMeasurements ? 'Available' : 'Estimated'}
` : ''}
**Multiple Wounds:** ${detectionInfo.multipleWounds ? 'Yes' : 'No'}`;
    }
    
    return `${disclaimer}\n\n${carePlan}${detectionSystemInfo}`;
    
  } catch (error: any) {
    console.error('Care plan generation error:', error);
    throw new Error(`Care plan generation failed: ${error.message}`);
  }
}
