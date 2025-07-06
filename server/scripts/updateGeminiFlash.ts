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
    
    // Update the model with warning about medical images
    await storage.updateAiAnalysisModel(flashModel.id!, {
      description: "Fast AI model - CAUTION: Has stricter content filters that block medical images. Use Gemini Pro for medical analysis.",
      isEnabled: false,
      capabilities: [...(flashModel.capabilities || []), "medical_limitation"],
      config: {
        ...flashModel.config,
        medical_warning: "Content filters may block medical images"
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