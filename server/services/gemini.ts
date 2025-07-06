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
        safetySettings: [
          {
            category: 'HARM_CATEGORY_HARASSMENT',
            threshold: 'BLOCK_ONLY_HIGH',
          },
          {
            category: 'HARM_CATEGORY_HATE_SPEECH',
            threshold: 'BLOCK_ONLY_HIGH',
          },
          {
            category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
            threshold: 'BLOCK_ONLY_HIGH',
          },
          {
            category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
            threshold: 'BLOCK_ONLY_HIGH',
          },
        ],
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
    "woundType": "type of wound (e.g., pressure ulcer, diabetic foot ulcer, surgical wound, etc.)",
    "stage": "stage if applicable (e.g., Stage 1, Stage 2, etc.)",
    "size": "small, medium, or large",
    "woundBed": "condition of wound bed (e.g., granulating, necrotic, sloughy, epithelializing)",
    "exudate": "none, low, moderate, or heavy",
    "infectionSigns": "array of observed signs (e.g., erythema, odor, increased warmth)",
    "location": "anatomical location",
    "additionalObservations": "any other relevant clinical observations",
    "confidence": "confidence score from 0.0 to 1.0 representing diagnostic certainty"
  }

  CONFIDENCE SCORING:
  - 0.9-1.0: Highly confident - clear visual indicators, typical presentation
  - 0.7-0.8: Moderately confident - good visual clarity, some uncertainty in classification
  - 0.5-0.6: Low confidence - poor image quality, atypical presentation, or multiple possibilities
  - 0.0-0.4: Very uncertain - insufficient visual information for reliable diagnosis

Please provide only the JSON response without any additional text or formatting.`;

  const response = await callGemini(model, prompt, imageBase64);
  
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
    console.error('Failed to parse Gemini JSON response:', jsonStr);
    throw new Error('Failed to parse wound assessment data from Gemini');
  }
}