import { GoogleGenAI } from "@google/genai";

// the newest Gemini model series is "gemini-2.5-flash" or gemini-2.5-pro"
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "" });

export async function callGemini(model: string, prompt: string, imageBase64?: string): Promise<string> {
  if (!["gemini-2.5-flash", "gemini-2.5-pro"].includes(model)) {
    throw new Error("Invalid Gemini model selection");
  }

  if (!process.env.GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY is not configured");
  }

  try {
    console.log(`Calling Gemini model: ${model}`);
    
    const parts = [];
    
    if (imageBase64) {
      // Remove data URL prefix if present
      const base64Data = imageBase64.replace(/^data:image\/[a-z]+;base64,/, '');
      console.log(`Adding image data, size: ${base64Data.length} characters`);
      parts.push({
        inlineData: {
          data: base64Data,
          mimeType: "image/jpeg",
        },
      });
    }
    
    parts.push({ text: prompt });
    console.log(`Sending request to Gemini with ${parts.length} parts`);

    let result;
    try {
      console.log(`About to call generateContent with model: ${model}`);
      result = await ai.models.generateContent({
        model,
        contents: parts,

      });
      console.log('Gemini API call successful');
    } catch (apiError: any) {
      console.error('Gemini API call failed:', apiError);
      console.error('API Error message:', apiError.message);
      console.error('API Error details:', JSON.stringify(apiError, null, 2));
      throw apiError;
    }

    console.log('Gemini result structure:', Object.keys(result));
    console.log('Gemini result type:', typeof result);
    console.log('Gemini result:', result);

    // Check for content blocking
    if (result.promptFeedback?.blockReason) {
      console.error('Gemini content blocked:', result.promptFeedback);
      throw new Error(`Gemini blocked request: ${result.promptFeedback.blockReason}. This may be due to content safety filters for medical images. Try using Gemini Pro instead.`);
    }

    const text = result.text;

    if (!text) {
      console.error('Gemini returned empty response');
      console.error('Result text property:', text);
      console.error('Result object:', JSON.stringify(result, null, 2));
      
      // Check if we have candidates but no text
      if (result.candidates && result.candidates.length > 0) {
        const candidate = result.candidates[0];
        console.error('Candidate structure:', Object.keys(candidate));
        console.error('Candidate:', candidate);
        
        if (candidate.finishReason) {
          throw new Error(`Gemini response blocked: ${candidate.finishReason}`);
        }
      }
      throw new Error("No response from Gemini");
    }

    console.log(`Gemini response received, length: ${text.length}`);
    return text;
  } catch (error: any) {
    console.error('Gemini API error details:', error);
    console.error('Error type:', typeof error);
    console.error('Error message:', error.message);
    console.error('Error stack:', error.stack);
    
    if (error.message?.includes('API key')) {
      throw new Error('Invalid or missing Gemini API key');
    }
    
    throw new Error(`Gemini processing failed: ${error.message}`);
  }
}

export async function analyzeWoundImageWithGemini(imageBase64: string, model: string, agentInstructions: string = ''): Promise<any> {
  const basePrompt = agentInstructions ? 
    `${agentInstructions}\n\nAnalyze this wound image and provide a detailed assessment in JSON format with the following structure:` :
    `You are a medical AI assistant specializing in wound assessment. Analyze this wound image and provide a detailed assessment in JSON format with the following structure:`;
    
  const prompt = `${basePrompt}
  {
    "woundType": "primary wound type (e.g., pressure ulcer, diabetic foot ulcer, surgical wound, etc.)",
    "stage": "stage if applicable (e.g., Stage 1, Stage 2, etc.)",
    "size": "small, medium, or large",
    "woundBed": "condition of wound bed (e.g., granulating, necrotic, sloughy, epithelializing)",
    "exudate": "none, low, moderate, or heavy",
    "infectionSigns": "array of observed signs (e.g., erythema, odor, increased warmth)",
    "location": "anatomical location",
    "additionalObservations": "any other relevant clinical observations",
    "confidence": "confidence score from 0.0 to 1.0 representing diagnostic certainty for primary diagnosis",
    "reasoning": "detailed explanation of visual indicators and clinical reasoning that led to this classification",
    "differentialDiagnosis": {
      "possibleTypes": [
        {
          "woundType": "primary diagnosis",
          "confidence": "0.0-1.0",
          "reasoning": "specific visual indicators supporting this diagnosis"
        },
        {
          "woundType": "secondary possibility",
          "confidence": "0.0-1.0", 
          "reasoning": "specific visual indicators supporting this diagnosis"
        }
      ],
      "questionsToAsk": [
        "targeted question to differentiate between possibilities",
        "another specific question based on visual findings"
      ]
    }
  }

  CONFIDENCE SCORING:
  - 0.9-1.0: Highly confident - clear visual indicators, typical presentation
  - 0.7-0.8: Moderately confident - good visual clarity, some uncertainty in classification
  - 0.5-0.6: Low confidence - poor image quality, atypical presentation, or multiple possibilities
  - 0.0-0.4: Very uncertain - insufficient visual information for reliable diagnosis

  DIFFERENTIAL DIAGNOSIS REQUIREMENTS (MANDATORY):
  - ALWAYS include at least 2-3 possible wound types with their confidence percentages
  - Consider anatomical location, wound characteristics, and typical presentations
  - For foot/heel wounds, consider: pressure ulcer, diabetic ulcer, venous ulcer, arterial ulcer
  - For leg wounds, consider: venous ulcer, arterial ulcer, traumatic wound
  - For pressure points, consider: pressure ulcer, but also diabetic complications if on feet
  - Provide specific targeted questions that would help distinguish between the possibilities
  - Questions should focus on medical history, mobility, diabetes status, circulation, etc.

  CRITICAL: You MUST include the "differentialDiagnosis" object with at least 2 possible wound types, even if you are confident in the primary diagnosis. This is a medical requirement for proper differential diagnosis.

Please provide only the JSON response without any additional text or formatting.`;

  const response = await callGemini(model, prompt, imageBase64);
  
  console.log('Gemini response preview:', response.substring(0, 500));
  
  // Clean up the response to extract JSON
  let jsonStr = response.trim();
  if (jsonStr.startsWith('```json')) {
    jsonStr = jsonStr.replace(/```json\s*/, '').replace(/```\s*$/, '');
  } else if (jsonStr.startsWith('```')) {
    jsonStr = jsonStr.replace(/```\s*/, '').replace(/```\s*$/, '');
  }
  
  try {
    return JSON.parse(jsonStr);
  } catch (parseError) {
    console.error('First JSON parse failed:', parseError.message);
    console.error('Raw response (first 1000 chars):', response.substring(0, 1000));
    
    // Try to extract JSON from text that might have explanatory content
    const jsonMatch = jsonStr.match(/\{[\s\S]*?\}/);
    if (jsonMatch) {
      try {
        console.log('Attempting to parse extracted JSON:', jsonMatch[0].substring(0, 200));
        return JSON.parse(jsonMatch[0]);
      } catch (secondParseError) {
        console.error('Second JSON parse failed:', secondParseError.message);
        console.error('Extracted JSON:', jsonMatch[0].substring(0, 500));
        throw new Error('Failed to parse wound assessment data from Gemini');
      }
    }
    
    console.error('No JSON found in response');
    throw new Error('Failed to parse wound assessment data from Gemini');
  }
}

export async function analyzeMultipleWoundImagesWithGemini(images: Array<{base64: string, mimeType: string}>, model: string, agentInstructions: string = ''): Promise<any> {
  const basePrompt = agentInstructions ? 
    `${agentInstructions}\n\nAnalyze the multiple wound images and provide a structured assessment. IMPORTANT: If images show different wounds, clearly identify this and focus on the primary wound for assessment.` :
    "You are a medical AI assistant specializing in wound assessment. Analyze the multiple wound images and provide a structured assessment. IMPORTANT: If images show different wounds, clearly identify this and focus on the primary wound for assessment.";

  const prompt = `${basePrompt}

Analyze these ${images.length} wound images and provide a detailed assessment in JSON format with the following structure:
{
  "multipleWounds": "boolean - true if images show different wounds/locations",
  "woundType": "type of primary wound (e.g., pressure ulcer, diabetic foot ulcer, surgical wound, etc.)",
  "stage": "stage if applicable (e.g., Stage 1, Stage 2, etc.)",
  "size": "small, medium, or large",
  "woundBed": "condition of wound bed (e.g., granulating, necrotic, sloughy, epithelializing)",
  "exudate": "none, low, moderate, or heavy",
  "infectionSigns": "array of observed signs (e.g., erythema, odor, increased warmth)",
  "location": "anatomical location of primary wound",
  "additionalObservations": "any other relevant clinical observations, including notes about multiple wounds if present",
  "confidence": "confidence score from 0.0 to 1.0 representing diagnostic certainty",
  "reasoning": "detailed explanation of visual indicators and clinical reasoning that led to this classification",
  "imageAnalysis": "detailed analysis of what each image shows and relationships between them"
}

CRITICAL INSTRUCTIONS:
- If images show different wounds in different locations, set "multipleWounds": true
- Focus your assessment on the most significant/primary wound
- In "additionalObservations", clearly state if multiple wounds are present
- In "imageAnalysis", describe what each image shows and whether they're the same wound from different angles or different wounds entirely
- Don't be afraid to say when images appear to show different wounds - accuracy is more important than convenience

CONFIDENCE SCORING:
- 0.9-1.0: Highly confident - clear visual indicators, typical presentation
- 0.7-0.8: Moderately confident - good visual clarity, some uncertainty in classification
- 0.5-0.6: Low confidence - poor image quality, atypical presentation, or multiple possibilities
- 0.0-0.4: Very uncertain - insufficient visual information for reliable diagnosis

Return ONLY valid JSON, no additional text.`;

  try {
    const parts = [];
    
    // Add all images
    for (const img of images) {
      const base64Data = img.base64.replace(/^data:image\/[a-z]+;base64,/, '');
      parts.push({
        inlineData: {
          data: base64Data,
          mimeType: img.mimeType,
        },
      });
    }
    
    parts.push({ text: prompt });

    const result = await ai.models.generateContent({
      model,
      contents: parts,
    });

    if (!result.candidates || result.candidates.length === 0) {
      throw new Error('No response from Gemini API');
    }

    const candidate = result.candidates[0];
    if (!candidate?.content?.parts?.[0]?.text) {
      throw new Error('Empty response from Gemini API');
    }

    const text = candidate.content.parts[0].text;
    let jsonStr = text.trim();
    
    // Remove JSON markers if present
    if (jsonStr.startsWith('```json')) {
      jsonStr = jsonStr.replace(/```json\s*/, '').replace(/```\s*$/, '');
    } else if (jsonStr.startsWith('```')) {
      jsonStr = jsonStr.replace(/```\s*/, '').replace(/```\s*$/, '');
    }
    
    return JSON.parse(jsonStr);
  } catch (error: any) {
    console.error('Gemini multiple image analysis error:', error);
    throw new Error(`Multiple image analysis failed: ${error.message}`);
  }
}