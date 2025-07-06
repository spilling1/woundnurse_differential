import { storage } from "../storage";
import type { InsertDetectionModel } from "@shared/schema";

const defaultDetectionModels: InsertDetectionModel[] = [
  {
    name: "yolo9",
    displayName: "YOLO9 Detection",
    description: "Primary YOLO9 wound detection service with precise measurements and boundary detection",
    isEnabled: true,
    priority: 100,
    endpoint: "http://localhost:8081/detect",
    requiresApiKey: false,
    modelType: "yolo",
    capabilities: ["detection", "measurements", "precise_boundaries"],
    config: {
      confidence_threshold: 0.5,
      include_measurements: true,
      detect_reference_objects: true,
      timeout: 30000
    }
  },
  {
    name: "google-cloud-vision",
    displayName: "Google Cloud Vision",
    description: "Google Cloud Vision API for reliable object detection and wound analysis",
    isEnabled: true,
    priority: 80,
    endpoint: "https://vision.googleapis.com/v1/images:annotate",
    requiresApiKey: true,
    apiKeyName: "GOOGLE_CLOUD_VISION_API_KEY",
    modelType: "cloud-api",
    capabilities: ["detection", "object_recognition", "medical_analysis"],
    config: {
      features: ["OBJECT_LOCALIZATION", "LABEL_DETECTION"],
      maxResults: 10
    }
  },
  {
    name: "azure-computer-vision",
    displayName: "Azure Computer Vision",
    description: "Microsoft Azure Computer Vision API for advanced image analysis",
    isEnabled: true,
    priority: 75,
    endpoint: null, // Set dynamically from AZURE_COMPUTER_VISION_ENDPOINT
    requiresApiKey: true,
    apiKeyName: "AZURE_COMPUTER_VISION_KEY",
    modelType: "cloud-api",
    capabilities: ["detection", "object_recognition", "medical_analysis"],
    config: {
      visualFeatures: ["Objects", "Tags"],
      details: ["Landmarks"],
      language: "en"
    }
  },
  {
    name: "cnn-classifier",
    displayName: "CNN Wound Classifier",
    description: "Custom trained CNN model for wound type classification (currently disabled due to accuracy concerns)",
    isEnabled: false,
    priority: 60,
    endpoint: null,
    requiresApiKey: false,
    modelType: "cnn",
    capabilities: ["classification", "wound_type_detection"],
    config: {
      model_path: "wound_classification_model.pth",
      confidence_threshold: 0.7,
      image_size: 224
    }
  },
  {
    name: "enhanced-fallback",
    displayName: "Enhanced Fallback Detection",
    description: "Intelligent fallback detection using basic image analysis when other methods fail",
    isEnabled: true,
    priority: 10,
    endpoint: null,
    requiresApiKey: false,
    modelType: "fallback",
    capabilities: ["basic_detection", "fallback_analysis"],
    config: {
      default_confidence: 0.5,
      center_detection: true,
      estimated_scale: 0.1
    }
  }
];

export async function seedDetectionModels(): Promise<void> {
  console.log("Seeding detection models...");
  
  try {
    // Check if models already exist
    const existingModels = await storage.getAllDetectionModels();
    
    if (existingModels.length > 0) {
      console.log(`Found ${existingModels.length} existing detection models. Skipping seed.`);
      return;
    }
    
    // Create default models
    for (const model of defaultDetectionModels) {
      try {
        await storage.createDetectionModel(model);
        console.log(`✓ Created detection model: ${model.displayName}`);
      } catch (error) {
        console.error(`✗ Failed to create detection model ${model.displayName}:`, error);
      }
    }
    
    console.log("Detection models seeded successfully!");
  } catch (error) {
    console.error("Error seeding detection models:", error);
    throw error;
  }
}

// Run seeding if this file is executed directly
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

if (import.meta.url === `file://${process.argv[1]}`) {
  seedDetectionModels()
    .then(() => {
      console.log("Seeding completed!");
      process.exit(0);
    })
    .catch((error) => {
      console.error("Seeding failed:", error);
      process.exit(1);
    });
}