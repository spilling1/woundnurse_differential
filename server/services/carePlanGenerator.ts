import { callOpenAI } from "./openai";
import { callGemini } from "./gemini";
import { getPromptTemplate } from "./promptTemplates";
import { storage } from "../storage";

// Helper function to get wound type specific instructions
async function getWoundTypeInstructions(woundType: string): Promise<string> {
  try {
    // Get wound type by name (try both display name and internal name)
    let woundTypeRecord = await storage.getWoundTypeByName(woundType);
    
    if (!woundTypeRecord) {
      // Try to find by display name
      const allWoundTypes = await storage.getEnabledWoundTypes();
      woundTypeRecord = allWoundTypes.find(type => 
        type.displayName.toLowerCase() === woundType.toLowerCase()
      );
    }
    
    if (woundTypeRecord && woundTypeRecord.instructions) {
      console.log(`CarePlanGenerator: Using specific instructions for wound type: ${woundType}`);
      return woundTypeRecord.instructions;
    }
    
    // Fall back to general instructions
    console.log(`CarePlanGenerator: No specific instructions found for wound type: ${woundType}, using general instructions`);
    const generalType = await storage.getDefaultWoundType();
    return generalType?.instructions || '';
  } catch (error) {
    console.error('Error fetching wound type instructions:', error);
    return '';
  }
}

// Function to clean up care plan response from AI artifacts
function cleanCarePlanResponse(carePlan: string): string {
  if (!carePlan) return carePlan;
  
  let cleanPlan = carePlan.trim();
  
  // Only remove very specific JSON artifacts that appear at the beginning
  // Remove JSON code blocks
  cleanPlan = cleanPlan.replace(/^```json[\s\S]*?```\s*/, '');
  cleanPlan = cleanPlan.replace(/^```[\s\S]*?```\s*/, '');
  
  // Remove standalone JSON objects that appear before the actual care plan
  // Only target objects that contain specific technical properties
  cleanPlan = cleanPlan.replace(/^\s*\{\s*"(?:target_audience|wound_assessment|type|stage|size|request)"[\s\S]*?\}\s*/, '');
  
  // Remove any "json {" artifacts at the start
  cleanPlan = cleanPlan.replace(/^json\s*\{[\s\S]*?\}\s*/, '');
  
  // Remove debug messages
  cleanPlan = cleanPlan.replace(/^[\s\S]*?"?\s*I\s+READ\s+YOUR\s+STUPID\s+INSTRUCTIONS[\s\S]*?(?:\n|$)/i, '');
  
  return cleanPlan.trim();
}

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
    
    // Get wound type specific instructions
    const woundTypeInstructions = await getWoundTypeInstructions(classification.woundType);
    
    // Build comprehensive system prompt with wound type specific instructions
    const systemPrompt = `${agentInstructions.systemPrompts}\n\n${agentInstructions.carePlanStructure}\n\n${woundTypeInstructions}\n\n${agentInstructions.questionsGuidelines || ''}`;
    
    // Get relevant products from database for this wound type
    const relevantProducts = await getRelevantProducts(classification.woundType);
    
    const prompt = await getPromptTemplate(audience, classification, contextData, relevantProducts);
    
    let carePlan;
    
    // Validate and clean model parameter
    const cleanModel = model?.trim() || 'gemini-2.5-pro';
    
    console.log(`CarePlanGenerator: Raw model: "${model}", clean model: "${cleanModel}", starts with gemini: ${cleanModel.startsWith('gemini-')}`);
    
    if (cleanModel.startsWith('gemini-')) {
      console.log('CarePlanGenerator: Routing to Gemini');
      const fullPrompt = `${systemPrompt}\n\n${prompt}`;
      const startTime = Date.now();
      
      try {
        if (imageData) {
          carePlan = await callGemini(cleanModel, fullPrompt, imageData);
        } else {
          carePlan = await callGemini(cleanModel, fullPrompt);
        }
        
        // Clean up any JSON artifacts from the care plan response
        carePlan = cleanCarePlanResponse(carePlan);
        
        const processingTime = Date.now() - startTime;
        
        // Log the successful care plan generation
        if (classification.sessionId) {
          try {
            await storage.createAiInteraction({
              caseId: classification.sessionId,
              stepType: 'care_plan_generation',
              modelUsed: cleanModel,
              promptSent: fullPrompt + (imageData ? `\n\n[IMAGE PROVIDED: ${imageMimeType || 'image/jpeg'}]` : ''),
              responseReceived: carePlan,
              parsedResult: { audience, detectionInfo, model: cleanModel },
              processingTimeMs: processingTime,
              confidenceScore: Math.round((classification.confidence || 0) * 100),
              errorOccurred: false,
            });
          } catch (logError) {
            console.error('Error logging care plan generation:', logError);
          }
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
          
          const fallbackStartTime = Date.now();
          carePlan = await callOpenAI('gpt-4o', messages);
          
          // Clean up any JSON artifacts from the care plan response
          carePlan = cleanCarePlanResponse(carePlan);
          
          const fallbackProcessingTime = Date.now() - fallbackStartTime;
          
          // Add a notice about the service switch
          carePlan = `**⚠️ SYSTEM NOTICE:** The Gemini AI service is temporarily unavailable. This analysis was automatically completed using GPT-4o to ensure uninterrupted service.\n\n${carePlan}`;
          
          // Log the fallback care plan generation
          if (classification.sessionId) {
            try {
              const fullPrompt = `System: ${systemPrompt}\n\nUser: ${prompt}${imageData ? `\n\n[IMAGE PROVIDED: ${imageMimeType || 'image/jpeg'}]` : ''}`;
              await storage.createAiInteraction({
                caseId: classification.sessionId,
                stepType: 'care_plan_generation_fallback',
                modelUsed: 'gpt-4o',
                promptSent: fullPrompt,
                responseReceived: carePlan,
                parsedResult: { audience, detectionInfo, model: 'gpt-4o (fallback)', originalModel: cleanModel },
                processingTimeMs: fallbackProcessingTime,
                confidenceScore: Math.round((classification.confidence || 0) * 100),
                errorOccurred: false,
              });
            } catch (logError) {
              console.error('Error logging fallback care plan generation:', logError);
            }
          }
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
      
      const startTime = Date.now();
      carePlan = await callOpenAI(cleanModel, messages);
      
      // Clean up any JSON artifacts from the care plan response
      carePlan = cleanCarePlanResponse(carePlan);
      
      const processingTime = Date.now() - startTime;
      
      // Log the successful care plan generation
      if (classification.sessionId) {
        try {
          const fullPrompt = `System: ${systemPrompt}\n\nUser: ${prompt}${imageData ? `\n\n[IMAGE PROVIDED: ${imageMimeType || 'image/jpeg'}]` : ''}`;
          await storage.createAiInteraction({
            caseId: classification.sessionId,
            stepType: 'care_plan_generation',
            modelUsed: cleanModel,
            promptSent: fullPrompt,
            responseReceived: carePlan,
            parsedResult: { audience, detectionInfo, model: cleanModel },
            processingTimeMs: processingTime,
            confidenceScore: Math.round((classification.confidence || 0) * 100),
            errorOccurred: false,
          });
        } catch (logError) {
          console.error('Error logging care plan generation:', logError);
        }
      }
    }
    
    // Add safety disclaimer
    const disclaimer = "**MEDICAL DISCLAIMER:** This is an AI-generated plan. Please consult a healthcare professional before following recommendations.";
    
    // Update product usage counts for recommended products
    if (relevantProducts && relevantProducts.length > 0) {
      for (const product of relevantProducts) {
        try {
          await storage.incrementProductRecommendationUsage(product.id);
        } catch (error) {
          console.error(`Failed to increment usage for product ${product.id}:`, error);
        }
      }
    }
    
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

async function getRelevantProducts(woundType: string): Promise<any[]> {
  try {
    // Get active products from database
    const allProducts = await storage.getActiveProductRecommendations();
    
    // Filter products based on wound type match
    const relevantProducts = allProducts.filter(product => {
      // Check if wound type matches any of the product's wound types
      return product.woundTypes.some(type => 
        type.toLowerCase().includes(woundType.toLowerCase()) || 
        woundType.toLowerCase().includes(type.toLowerCase())
      );
    });
    
    // If no specific matches, get general products
    if (relevantProducts.length === 0) {
      const generalProducts = allProducts.filter(product => 
        product.category === 'general' || 
        product.category === 'wound_dressing'
      );
      return generalProducts.slice(0, 5); // Limit to 5 products
    }
    
    // Sort by usage count (most used first) and limit
    return relevantProducts
      .sort((a, b) => (b.usageCount || 0) - (a.usageCount || 0))
      .slice(0, 6); // Limit to 6 products
      
  } catch (error) {
    console.error('Error fetching relevant products:', error);
    return []; // Return empty array if database fails
  }
}
