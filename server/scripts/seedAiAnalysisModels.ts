import { storage } from "../storage";
import type { InsertAiAnalysisModel } from "@shared/schema";

const defaultAiAnalysisModels: InsertAiAnalysisModel[] = [
  {
    name: "gemini-2.5-pro",
    displayName: "Gemini 2.5 Pro",
    description: "Google's most advanced AI model with superior reasoning capabilities and medical analysis",
    isEnabled: true,
    isDefault: true, // Default model
    priority: 100,
    provider: "google",
    modelId: "gemini-2.5-pro",
    requiresApiKey: true,
    apiKeyName: "GOOGLE_AI_API_KEY",
    capabilities: ["vision", "text", "reasoning", "medical_analysis"],
    config: {
      temperature: 0.1,
      max_tokens: 4000,
      safety_settings: {
        harassment: "BLOCK_MEDIUM_AND_ABOVE",
        hate_speech: "BLOCK_MEDIUM_AND_ABOVE",
        sexually_explicit: "BLOCK_MEDIUM_AND_ABOVE",
        dangerous_content: "BLOCK_MEDIUM_AND_ABOVE"
      }
    }
  },
  {
    name: "gemini-2.5-flash",
    displayName: "Gemini 2.5 Flash",
    description: "Fast and efficient AI model for quick wound analysis with good accuracy",
    isEnabled: true,
    isDefault: false,
    priority: 90,
    provider: "google",
    modelId: "gemini-2.5-flash",
    requiresApiKey: true,
    apiKeyName: "GOOGLE_AI_API_KEY",
    capabilities: ["vision", "text", "reasoning"],
    config: {
      temperature: 0.1,
      max_tokens: 3000,
      safety_settings: {
        harassment: "BLOCK_MEDIUM_AND_ABOVE",
        hate_speech: "BLOCK_MEDIUM_AND_ABOVE",
        sexually_explicit: "BLOCK_MEDIUM_AND_ABOVE",
        dangerous_content: "BLOCK_MEDIUM_AND_ABOVE"
      }
    }
  },
  {
    name: "gpt-4o",
    displayName: "GPT-4o",
    description: "OpenAI's advanced multimodal model with excellent vision and reasoning capabilities",
    isEnabled: true,
    isDefault: false,
    priority: 85,
    provider: "openai",
    modelId: "gpt-4o",
    requiresApiKey: true,
    apiKeyName: "OPENAI_API_KEY",
    capabilities: ["vision", "text", "reasoning", "detailed_analysis"],
    config: {
      temperature: 0.1,
      max_tokens: 4000,
      frequency_penalty: 0,
      presence_penalty: 0
    }
  },
  {
    name: "gpt-3.5-turbo",
    displayName: "GPT-3.5 Turbo",
    description: "Cost-effective OpenAI model for basic wound analysis and care plan generation",
    isEnabled: true,
    isDefault: false,
    priority: 70,
    provider: "openai",
    modelId: "gpt-3.5-turbo",
    requiresApiKey: true,
    apiKeyName: "OPENAI_API_KEY",
    capabilities: ["text", "reasoning"],
    config: {
      temperature: 0.1,
      max_tokens: 3000,
      frequency_penalty: 0,
      presence_penalty: 0
    }
  },
  {
    name: "gpt-3.5-pro",
    displayName: "GPT-3.5 Pro",
    description: "Enhanced GPT-3.5 with improved medical knowledge and analysis capabilities",
    isEnabled: true,
    isDefault: false,
    priority: 75,
    provider: "openai",
    modelId: "gpt-3.5-turbo",
    requiresApiKey: true,
    apiKeyName: "OPENAI_API_KEY",
    capabilities: ["text", "reasoning", "medical_knowledge"],
    config: {
      temperature: 0.05,
      max_tokens: 3500,
      frequency_penalty: 0.1,
      presence_penalty: 0.1,
      enhanced_prompts: true
    }
  }
];

export async function seedAiAnalysisModels(): Promise<void> {
  console.log("Seeding AI analysis models...");
  
  try {
    // Check if models already exist
    const existingModels = await storage.getAllAiAnalysisModels();
    
    if (existingModels.length > 0) {
      console.log(`Found ${existingModels.length} existing AI analysis models. Skipping seed.`);
      return;
    }
    
    // Create default models
    for (const model of defaultAiAnalysisModels) {
      try {
        await storage.createAiAnalysisModel(model);
        console.log(`✓ Created AI analysis model: ${model.displayName}`);
      } catch (error) {
        console.error(`✗ Failed to create AI analysis model ${model.displayName}:`, error);
      }
    }
    
    console.log("AI analysis models seeded successfully!");
  } catch (error) {
    console.error("Error seeding AI analysis models:", error);
    throw error;
  }
}

// Run seeding if this file is executed directly
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

if (process.argv[1] === __filename) {
  seedAiAnalysisModels()
    .then(() => {
      console.log("Seeding completed successfully!");
      process.exit(0);
    })
    .catch((error) => {
      console.error("Seeding failed:", error);
      process.exit(1);
    });
}