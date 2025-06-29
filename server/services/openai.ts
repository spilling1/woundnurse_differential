import OpenAI from "openai";

// the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
const openai = new OpenAI({ 
  apiKey: process.env.OPENAI_API_KEY || process.env.OPENAI_API_KEY_ENV_VAR || ""
});

export async function callOpenAI(model: string, messages: any[], responseFormat: any = null): Promise<string> {
  if (!["gpt-4o", "gpt-3.5", "gpt-3.5-pro"].includes(model)) {
    throw new Error("Invalid OpenAI model selection");
  }

  try {
    const params: any = {
      model: model === "gpt-3.5-pro" ? "gpt-3.5-turbo" : model,
      messages,
      max_tokens: 1000,
    };

    if (responseFormat) {
      params.response_format = responseFormat;
    }

    console.log('OpenAI API call params:', JSON.stringify(params, null, 2));
    const response = await openai.chat.completions.create(params);
    console.log('OpenAI API response:', JSON.stringify(response, null, 2));
    
    if (!response.choices[0]?.message?.content) {
      console.error('OpenAI response missing content:', response.choices[0]);
      throw new Error("No response from OpenAI");
    }

    return response.choices[0].message.content;
  } catch (error: any) {
    console.error('OpenAI API error:', error);
    throw new Error(`AI processing failed: ${error.message}`);
  }
}

export async function analyzeWoundImage(imageBase64: string, model: string, mimeType: string = 'image/jpeg'): Promise<any> {
  const messages = [
    {
      role: "system",
      content: "You are a medical AI assistant specializing in wound assessment. Analyze the wound image and provide a structured assessment in JSON format."
    },
    {
      role: "user",
      content: [
        {
          type: "text",
          text: `Analyze this wound image and provide a detailed assessment in JSON format with the following structure:
          {
            "woundType": "type of wound (e.g., pressure ulcer, diabetic foot ulcer, surgical wound, etc.)",
            "stage": "stage if applicable (e.g., Stage 1, Stage 2, etc.)",
            "size": "small, medium, or large",
            "woundBed": "condition of wound bed (e.g., granulating, necrotic, sloughy, epithelializing)",
            "exudate": "none, low, moderate, or heavy",
            "infectionSigns": "array of observed signs (e.g., erythema, odor, increased warmth)",
            "location": "anatomical location",
            "additionalObservations": "any other relevant clinical observations"
          }`
        },
        {
          type: "image_url",
          image_url: {
            url: `data:${mimeType};base64,${imageBase64}`
          }
        }
      ]
    }
  ];

  const response = await callOpenAI(model, messages, { type: "json_object" });
  return JSON.parse(response);
}
