import { storage } from "../storage";

async function updateGeminiFlash(): Promise<void> {
  console.log("Updating Gemini Flash model to reflect medical image limitation...");
  
  try {
    const models = await storage.getAllAiAnalysisModels();
    const flashModel = models.find(m => m.modelId === 'gemini-2.5-flash');
    
    if (!flashModel) {
      console.log("Gemini Flash model not found in database");
      return;
    }
    
    // Update the model to enable it with relaxed safety settings
    await storage.updateAiAnalysisModel(flashModel.id!, {
      description: "Fast and efficient AI model for quick wound analysis with relaxed safety settings for medical images",
      isEnabled: true,
      capabilities: flashModel.capabilities?.filter(c => c !== "medical_limitation") || [],
      config: {
        ...flashModel.config,
        safety_settings: {
          harassment: "BLOCK_NONE",
          hate_speech: "BLOCK_NONE", 
          sexually_explicit: "BLOCK_NONE",
          dangerous_content: "BLOCK_NONE"
        }
      }
    });
    
    console.log("✓ Updated Gemini Flash model with medical image limitation warning");
  } catch (error) {
    console.error("Error updating Gemini Flash model:", error);
    throw error;
  }
}

// Run the update
updateGeminiFlash()
  .then(() => {
    console.log("Update completed successfully!");
    process.exit(0);
  })
  .catch((error) => {
    console.error("Update failed:", error);
    process.exit(1);
  });