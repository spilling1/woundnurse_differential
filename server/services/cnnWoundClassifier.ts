import { spawn } from 'child_process';
import { writeFileSync, unlinkSync, readdirSync } from 'fs';
import { join } from 'path';

interface CNNClassificationResult {
  woundType: string;
  confidence: number;
  allProbabilities: Record<string, number>;
  processingTime: number;
  model: string;
}

export class CNNWoundClassifier {
  private static instance: CNNWoundClassifier;
  private isInitialized = false;
  private availableModels: string[] = [];

  private constructor() {}

  static getInstance(): CNNWoundClassifier {
    if (!CNNWoundClassifier.instance) {
      CNNWoundClassifier.instance = new CNNWoundClassifier();
    }
    return CNNWoundClassifier.instance;
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      // Check for available trained models
      try {
        const files = readdirSync(process.cwd());
        this.availableModels = files
          .filter((file: string) => file.endsWith('.pth'))
          .filter((file: string) => file.includes('wound_model') || file.includes('quick_wound_model') || file.includes('mini_'));
      } catch (readError) {
        console.log('Could not read directory for model files:', readError);
        this.availableModels = [];
      }

      console.log('CNN Classifier initialized. Available models:', this.availableModels);
      this.isInitialized = true;
    } catch (error) {
      console.error('Failed to initialize CNN classifier:', error);
      throw error as Error;
    }
  }

  async classifyWound(imageBase64: string): Promise<CNNClassificationResult> {
    await this.initialize();

    if (this.availableModels.length === 0) {
      throw new Error('No trained CNN models available. Please train models first.');
    }

    const startTime = Date.now();

    try {
      // Use the best available model (highest accuracy)
      const bestModel = this.selectBestModel();
      
      // Save base64 image to temporary file
      const tempImagePath = this.saveImageToTemp(imageBase64);
      
      try {
        // Run CNN prediction using Python script
        const result = await this.runCNNPrediction(tempImagePath, bestModel);
        
        const processingTime = Date.now() - startTime;

        return {
          woundType: result.predicted_class,
          confidence: result.confidence,
          allProbabilities: result.all_probabilities || {},
          processingTime,
          model: bestModel
        };
        
      } finally {
        // Clean up temporary file
        this.cleanupTempFile(tempImagePath);
      }
      
    } catch (error) {
      console.error('CNN classification error:', error);
      throw new Error(`CNN wound classification failed: ${error.message}`);
    }
  }

  private selectBestModel(): string {
    // Priority order: quick_wound_model (highest accuracy) > mini models > others
    const quickModels = this.availableModels.filter(m => m.includes('quick_wound_model'));
    if (quickModels.length > 0) {
      // Find highest accuracy quick model
      return quickModels.sort((a, b) => {
        const aAcc = this.extractAccuracy(a);
        const bAcc = this.extractAccuracy(b);
        return bAcc - aAcc;
      })[0];
    }

    const miniModels = this.availableModels.filter(m => m.includes('mini_'));
    if (miniModels.length > 0) {
      return miniModels.sort((a, b) => {
        const aAcc = this.extractAccuracy(a);
        const bAcc = this.extractAccuracy(b);
        return bAcc - aAcc;
      })[0];
    }

    return this.availableModels[0];
  }

  private extractAccuracy(filename: string): number {
    const match = filename.match(/acc_(\d+\.?\d*)/);
    return match ? parseFloat(match[1]) : 0;
  }

  private saveImageToTemp(imageBase64: string): string {
    const imageBuffer = Buffer.from(imageBase64.replace(/^data:image\/[a-z]+;base64,/, ''), 'base64');
    const tempPath = join(process.cwd(), `temp_wound_${Date.now()}.jpg`);
    writeFileSync(tempPath, imageBuffer);
    return tempPath;
  }

  private cleanupTempFile(filePath: string): void {
    try {
      unlinkSync(filePath);
    } catch (error) {
      console.warn('Failed to cleanup temp file:', filePath);
    }
  }

  private async runCNNPrediction(imagePath: string, modelPath: string): Promise<any> {
    return new Promise((resolve, reject) => {
      // Create a Python script to run the prediction
      const pythonScript = `
import torch
from PIL import Image
from torchvision import transforms
import json
import sys

class TinyCNN(torch.nn.Module):
    def __init__(self, num_classes=6):
        super(TinyCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(4, stride=4),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

def predict_image(image_path, model_path):
    # Load model
    device = torch.device('cpu')
    model = TinyCNN()
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except:
        print(json.dumps({"error": "Failed to load model"}))
        return
    
    # Class mapping
    class_names = [
        "background", "diabetic_ulcer", "neuropathic_ulcer", 
        "pressure_ulcer", "surgical_wound", "venous_ulcer"
    ]
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        
        # Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Prepare result
            result = {
                "predicted_class": class_names[predicted.item()],
                "confidence": confidence.item() * 100,
                "all_probabilities": {
                    class_names[i]: prob.item() * 100 
                    for i, prob in enumerate(probabilities[0])
                }
            }
            
            print(json.dumps(result))
            
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: python script.py <image_path> <model_path>"}))
        sys.exit(1)
    
    predict_image(sys.argv[1], sys.argv[2])
`;

      // Write Python script to temporary file
      const scriptPath = join(process.cwd(), `cnn_predict_${Date.now()}.py`);
      writeFileSync(scriptPath, pythonScript);

      try {
        const python = spawn('python', [scriptPath, imagePath, modelPath]);
        
        let stdout = '';
        let stderr = '';

        python.stdout.on('data', (data) => {
          stdout += data.toString();
        });

        python.stderr.on('data', (data) => {
          stderr += data.toString();
        });

        python.on('close', (code) => {
          // Cleanup script file
          try {
            unlinkSync(scriptPath);
          } catch (e) {}

          if (code !== 0) {
            reject(new Error(`Python process failed with code ${code}: ${stderr}`));
            return;
          }

          try {
            const result = JSON.parse(stdout.trim());
            if (result.error) {
              reject(new Error(result.error));
            } else {
              resolve(result);
            }
          } catch (error) {
            reject(new Error(`Failed to parse prediction result: ${stdout}`));
          }
        });

        python.on('error', (error) => {
          try {
            unlinkSync(scriptPath);
          } catch (e) {}
          reject(error);
        });

      } catch (error) {
        try {
          unlinkSync(scriptPath);
        } catch (e) {}
        reject(error);
      }
    });
  }

  async getModelInfo(): Promise<{ available: boolean; models: string[]; bestModel?: string }> {
    await this.initialize();
    
    return {
      available: this.availableModels.length > 0,
      models: this.availableModels,
      bestModel: this.availableModels.length > 0 ? this.selectBestModel() : undefined
    };
  }
}

export const cnnWoundClassifier = CNNWoundClassifier.getInstance();

// Helper function to convert CNN result to standard classification format
export function convertCNNToStandardClassification(cnnResult: CNNClassificationResult): any {
  // Log detailed probabilities for debugging
  console.log('CNN Detailed Results:', {
    predicted: cnnResult.woundType,
    confidence: cnnResult.confidence,
    allProbabilities: cnnResult.allProbabilities
  });
  
  // Map CNN wound types to standard wound classification
  const woundTypeMapping: Record<string, string> = {
    'background': 'No wound detected',
    'diabetic_ulcer': 'Diabetic foot ulcer',
    'neuropathic_ulcer': 'Neuropathic ulcer',
    'pressure_ulcer': 'Pressure ulcer',
    'surgical_wound': 'Surgical wound',
    'venous_ulcer': 'Venous leg ulcer'
  };
  
  // Check if any wound type has high probability even if not the top prediction
  const woundProbabilities = Object.entries(cnnResult.allProbabilities)
    .filter(([type, _]) => type !== 'background')
    .sort(([_, a], [__, b]) => b - a);
  
  // If background is predicted but a wound type has >40% probability, flag for review
  if (cnnResult.woundType === 'background' && woundProbabilities.length > 0) {
    const [topWoundType, topWoundProb] = woundProbabilities[0];
    if (topWoundProb > 40) {
      console.log(`CNN: Background predicted but ${topWoundType} has ${topWoundProb.toFixed(1)}% probability - may need review`);
    }
  }

  // Determine size category based on confidence and type
  const determineSize = (woundType: string, confidence: number): string => {
    if (woundType === 'background') return 'N/A';
    if (confidence > 90) return 'clearly defined';
    if (confidence > 70) return 'well-defined';
    return 'requires closer examination';
  };

  // Determine stage based on wound type and confidence
  const determineStage = (woundType: string, confidence: number): string => {
    if (woundType === 'background') return 'N/A';
    if (woundType === 'pressure_ulcer') {
      if (confidence > 85) return 'Stage II-III (requires clinical assessment)';
      return 'Requires clinical staging';
    }
    if (woundType === 'diabetic_ulcer') return 'Wagner grade assessment needed';
    return 'Assessment required';
  };

  return {
    woundType: woundTypeMapping[cnnResult.woundType] || cnnResult.woundType,
    stage: determineStage(cnnResult.woundType, cnnResult.confidence),
    size: determineSize(cnnResult.woundType, cnnResult.confidence),
    woundBed: cnnResult.woundType === 'background' ? 'N/A' : 'Visual assessment required',
    exudate: cnnResult.woundType === 'background' ? 'N/A' : 'Clinical assessment needed',
    infectionSigns: [],
    location: 'Image-based analysis',
    additionalObservations: `CNN Classification: ${cnnResult.woundType} (${cnnResult.confidence.toFixed(1)}% confidence)`,
    confidence: cnnResult.confidence / 100,
    cnnData: {
      model: cnnResult.model,
      processingTime: cnnResult.processingTime,
      allProbabilities: cnnResult.allProbabilities,
      machineLearning: true
    }
  };
}