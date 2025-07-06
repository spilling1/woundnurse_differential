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
    // Get AI instructions - must be configured
    const agentInstructions = await storage.getActiveAgentInstructions();
    if (!agentInstructions?.systemPrompts) {
      throw new Error('AI Configuration not found. Please configure AI system prompts in Settings before generating care plans.');
    }
    
    const systemPrompt = agentInstructions.systemPrompts;
    
    const prompt = await getPromptTemplate(audience, classification, contextData);
    
    let carePlan;
    
    // Validate and clean model parameter
    const cleanModel = model?.trim() || 'gemini-2.5-pro';
    
    console.log(`CarePlanGenerator: Raw model: "${model}", clean model: "${cleanModel}", starts with gemini: ${cleanModel.startsWith('gemini-')}`);
    
    if (cleanModel.startsWith('gemini-')) {
      console.log('CarePlanGenerator: Routing to Gemini');
      const fullPrompt = `${systemPrompt}\n\n${prompt}`;
      try {
        if (imageData) {
          carePlan = await callGemini(cleanModel, fullPrompt, imageData);
        } else {
          carePlan = await callGemini(cleanModel, fullPrompt);
        }
      } catch (geminiError: any) {
        // Check if this is a quota error
        if (geminiError.message?.includes('quota') || geminiError.message?.includes('RESOURCE_EXHAUSTED')) {
          console.log('CarePlanGenerator: Gemini service temporarily unavailable, automatically switching to GPT-4o');
          // Automatically switch to GPT-4o when Gemini service is unavailable
          // Add a notice to the care plan about the service switch
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
          
          carePlan = await callOpenAI('gpt-4o', messages);
          // Add a notice about the service switch
          carePlan = `**⚠️ SYSTEM NOTICE:** The Gemini AI service is temporarily unavailable. This analysis was automatically completed using GPT-4o to ensure uninterrupted service.\n\n${carePlan}`;
        } else {
          // Re-throw non-quota errors
          throw geminiError;
        }
      }
    } else {
      console.log('CarePlanGenerator: Routing to OpenAI');
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
      
      carePlan = await callOpenAI(cleanModel, messages);
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

**Detailed Detection Results:**
1. **Wound Type:** ${classification.woundType || 'N/A'}
2. **Location:** ${classification.location || 'N/A'}
3. **Size:** ${hasDetections?.measurements?.areaMm2 ? `${hasDetections.measurements.areaMm2}mm²` : 'N/A'}
4. **Depth:** ${classification.depth || 'N/A'}
5. **Exudate:** ${classification.exudate || 'N/A'}
6. **Wound Edges:** ${classification.woundEdges || 'N/A'}
7. **Signs of Infection:** ${classification.signsOfInfection || 'N/A'}

${hasDetections ? `
**Technical Measurements:**
- Confidence: ${Math.round((hasDetections.confidence || 0) * 100)}%
- Length: ${hasDetections.measurements?.lengthMm || 'N/A'}mm
- Width: ${hasDetections.measurements?.widthMm || 'N/A'}mm  
- Area: ${hasDetections.measurements?.areaMm2 || 'N/A'}mm²
- Scale Calibrated: ${hasDetections.scaleCalibrated ? 'Yes' : 'No'}
- Precise Measurements: ${classification.preciseMeasurements ? 'Available' : 'Estimated'}
` : `
**Technical Measurements:**
- Confidence: N/A
- Length: N/A
- Width: N/A
- Area: N/A
- Scale Calibrated: No
- Precise Measurements: Not evaluated
`}
**Multiple Wounds:** ${detectionInfo.multipleWounds ? 'Yes' : 'No'}`;
    }
    
    return `${disclaimer}\n\n${carePlan}${detectionSystemInfo}`;
    
  } catch (error: any) {
    console.error('Care plan generation error:', error);
    throw new Error(`Care plan generation failed: ${error.message}`);
  }
}
