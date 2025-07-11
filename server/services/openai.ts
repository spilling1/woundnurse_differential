import OpenAI from "openai";

// the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
const openai = new OpenAI({ 
  apiKey: process.env.OPENAI_API_KEY || process.env.OPENAI_API_KEY_ENV_VAR || ""
});

export async function callOpenAI(model: string, messages: any[], responseFormat: any = null): Promise<string> {
  // Normalize model name to lowercase
  const normalizedModel = model.toLowerCase();
  
  console.log(`OpenAI callOpenAI: model="${model}", normalized="${normalizedModel}"`);
  
  if (!["gpt-4o", "gpt-3.5", "gpt-3.5-pro"].includes(normalizedModel)) {
    console.error(`OpenAI model validation failed: "${model}" (normalized: "${normalizedModel}") not in allowed list`);
    throw new Error(`Invalid OpenAI model selection: ${model}. Supported models: gpt-4o, gpt-3.5, gpt-3.5-pro`);
  }

  try {
    const actualModel = normalizedModel === "gpt-3.5-pro" ? "gpt-3.5-turbo" : normalizedModel;
    console.log(`OpenAI API call: using model "${actualModel}", max_tokens: 4000`);
    
    const params: any = {
      model: actualModel,
      messages,
      max_tokens: 4000, // Increased for comprehensive care plans
    };

    if (responseFormat) {
      params.response_format = responseFormat;
    }

    const response = await openai.chat.completions.create(params);
    
    if (!response.choices[0]?.message?.content) {
      // Check if OpenAI refused the request
      if (response.choices[0]?.message?.refusal) {
        console.error('OpenAI refused request:', response.choices[0].message.refusal);
        throw new Error(`OpenAI refusal: ${response.choices[0].message.refusal}`);
      }
      console.error('OpenAI response missing content:', response.choices[0]);
      console.error('Full OpenAI response:', JSON.stringify(response, null, 2));
      throw new Error("No response from OpenAI");
    }

    console.log(`OpenAI API success: received ${response.choices[0].message.content.length} characters`);
    return response.choices[0].message.content;
  } catch (error: any) {
    console.error('OpenAI API error:', error);
    throw new Error(`AI processing failed: ${error.message}`);
  }
}

export async function analyzeWoundImage(imageBase64: string, model: string, mimeType: string = 'image/jpeg', agentInstructions: string = ''): Promise<any> {
  const systemMessage = agentInstructions ? 
    `${agentInstructions}\n\nAnalyze the wound image and provide a structured assessment in JSON format.` :
    "You are a medical AI assistant specializing in wound assessment. Analyze the wound image and provide a structured assessment in JSON format.";
    
  const messages = [
    {
      role: "system",
      content: systemMessage
    },
    {
      role: "user",
      content: [
        {
          type: "text",
          text: `Analyze this wound image and provide a detailed assessment in JSON format with the following structure:
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
          
          CRITICAL: You MUST include the "differentialDiagnosis" object with at least 2 possible wound types, even if you are confident in the primary diagnosis. This is a medical requirement for proper differential diagnosis.`
        },
        {
          type: "image_url",
          image_url: {
            url: `data:${mimeType};base64,${imageBase64}`,
            detail: "high"
          }
        }
      ]
    }
  ];

  const response = await callOpenAI(model, messages, { type: "json_object" });
  return JSON.parse(response);
}

export async function analyzeMultipleWoundImages(images: Array<{base64: string, mimeType: string}>, model: string, agentInstructions: string = ''): Promise<any> {
  const systemMessage = agentInstructions ? 
    `${agentInstructions}\n\nAnalyze the multiple wound images and provide a structured assessment. IMPORTANT: If images show different wounds, clearly identify this and focus on the primary wound for assessment.` :
    "You are a medical AI assistant specializing in wound assessment. Analyze the multiple wound images and provide a structured assessment. IMPORTANT: If images show different wounds, clearly identify this and focus on the primary wound for assessment.";
    
  const imageContents = images.map(img => ({
    type: "image_url",
    image_url: {
      url: `data:${img.mimeType};base64,${img.base64}`,
      detail: "high"
    }
  }));

  const messages = [
    {
      role: "system",
      content: systemMessage
    },
    {
      role: "user",
      content: [
        {
          type: "text",
          text: `Analyze these ${images.length} wound images and provide a detailed assessment in JSON format with the following structure:
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
          - 0.0-0.4: Very uncertain - insufficient visual information for reliable diagnosis`
        },
        ...imageContents
      ]
    }
  ];

  const response = await callOpenAI(model, messages, { type: "json_object" });
  return JSON.parse(response);
}
