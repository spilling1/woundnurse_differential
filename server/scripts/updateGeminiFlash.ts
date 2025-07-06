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
    
    // Update the model to reflect API-level medical image blocking
    await storage.updateAiAnalysisModel(flashModel.id!, {
      description: "Fast AI model - API blocks medical images even with relaxed safety settings. Use Gemini Pro for wound analysis.",
      isEnabled: false,
      capabilities: [...(flashModel.capabilities?.filter(c => c !== "medical_limitation") || []), "api_medical_restriction"],
      config: {
        ...flashModel.config,
        api_limitation: "Medical images blocked at API level regardless of safety settings"
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