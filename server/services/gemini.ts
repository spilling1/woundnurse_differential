import { GoogleGenAI } from "@google/genai";

// the newest Gemini model series is "gemini-2.5-flash" or gemini-2.5-pro"
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "" });

export async function callGemini(model: string, prompt: string, imageBase64?: string): Promise<string> {
  if (!["gemini-2.5-flash", "gemini-2.5-pro"].includes(model)) {
    throw new Error("Invalid Gemini model selection");
  }

  try {
    const contents = [];
    
    if (imageBase64) {
      contents.push({
        inlineData: {
          data: imageBase64,
          mimeType: "image/jpeg",
        },
      });
    }
    
    contents.push(prompt);

    const response = await ai.models.generateContent({
      model,
      contents: contents,
    });

    if (!response.text) {
      throw new Error("No response from Gemini");
    }

    return response.text;
  } catch (error: any) {
    console.error('Gemini API error:', error);
    throw new Error(`Gemini processing failed: ${error.message}`);
  }
}

export async function analyzeWoundImageWithGemini(imageBase64: string, model: string): Promise<any> {
  const prompt = `You are a medical AI assistant specializing in wound assessment. Analyze this wound image and provide a detailed assessment in JSON format with the following structure:
  {
    "woundType": "type of wound (e.g., pressure ulcer, diabetic foot ulcer, surgical wound, etc.)",
    "stage": "stage if applicable (e.g., Stage 1, Stage 2, etc.)",
    "size": "small, medium, or large",
    "woundBed": "condition of wound bed (e.g., granulating, necrotic, sloughy, epithelializing)",
    "exudate": "none, low, moderate, or heavy",
    "infectionSigns": "array of observed signs (e.g., erythema, odor, increased warmth)",
    "location": "anatomical location",
    "additionalObservations": "any other relevant clinical observations"
  }

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