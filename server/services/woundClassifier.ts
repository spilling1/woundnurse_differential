import { analyzeWoundImage } from "./openai";
import { analyzeWoundImageWithGemini } from "./gemini";
import { storage } from "../storage";
import { woundDetectionService } from "./woundDetection";
import { cnnWoundClassifier, convertCNNToStandardClassification } from "./cnnWoundClassifier";

export async function classifyWound(imageBase64: string, model: string, mimeType: string = 'image/jpeg'): Promise<any> {
  try {
    // Step 1: Try CNN-based wound classification first (priority method)
    let classification;
    let usedCNN = false;
    
    try {
      const cnnModelInfo = await cnnWoundClassifier.getModelInfo();
      
      if (cnnModelInfo.available) {
        console.log(`Using trained CNN model: ${cnnModelInfo.bestModel}`);
        const cnnResult = await cnnWoundClassifier.classifyWound(imageBase64);
        classification = convertCNNToStandardClassification(cnnResult);
        usedCNN = true;
        console.log(`CNN Classification: ${cnnResult.woundType} (${cnnResult.confidence.toFixed(1)}% confidence)`);
      } else {
        console.log('No trained CNN models available, falling back to AI vision models');
        throw new Error('CNN models not available');
      }
    } catch (cnnError) {
      console.log('CNN classification failed, using AI vision models as fallback:', (cnnError as Error).message);
      usedCNN = false;
    }
    
    // Step 2: Perform YOLO-based wound detection for measurements (regardless of classification method)
    const detectionResult = await woundDetectionService.detectWounds(imageBase64, mimeType);
    
    // Step 3: If CNN failed, use AI vision models as fallback
    if (!usedCNN) {
      // Get agent instructions from database to include in analysis
      const agentInstructions = await storage.getActiveAgentInstructions();
      const instructions = agentInstructions ? 
        `${agentInstructions.systemPrompts}\n\n${agentInstructions.carePlanStructure}\n\n${agentInstructions.specificWoundCare}\n\n${agentInstructions.questionsGuidelines || ''}` : '';
      
      // Enhance AI analysis with detection data
      const enhancedInstructions = `${instructions}

WOUND DETECTION DATA:
- Number of wounds detected: ${detectionResult.detections.length}
- Detection confidence: ${detectionResult.detections[0]?.confidence || 0}
${detectionResult.detections.length > 0 ? `
- Wound measurements: ${detectionResult.detections[0].measurements.lengthMm}mm x ${detectionResult.detections[0].measurements.widthMm}mm
- Wound area: ${detectionResult.detections[0].measurements.areaMm2}mm²
- Scale calibrated: ${detectionResult.detections[0].scaleCalibrated ? 'Yes' : 'No'}
- Reference object detected: ${detectionResult.detections[0].referenceObjectDetected ? 'Yes' : 'No'}
` : ''}
Use this detection data to improve your wound analysis accuracy.`;
      
      if (model.startsWith('gemini-')) {
        classification = await analyzeWoundImageWithGemini(imageBase64, model, enhancedInstructions);
      } else {
        classification = await analyzeWoundImage(imageBase64, model, mimeType, enhancedInstructions);
      }
    }
    
    // Step 3: Validate and normalize the classification
    const normalizedClassification = {
      woundType: classification.woundType || "Unspecified",
      stage: classification.stage || "Not determined",
      size: normalizeSize(classification.size),
      woundBed: classification.woundBed || "Not assessed",
      exudate: normalizeExudate(classification.exudate),
      infectionSigns: Array.isArray(classification.infectionSigns) 
        ? classification.infectionSigns 
        : [],
      location: classification.location || "Not specified",
      additionalObservations: classification.additionalObservations || "",
      confidence: classification.confidence || 0.5
    };

    // Step 4: Enhance classification with detection data
    const enhancedClassification = enhanceClassificationWithDetection(
      normalizedClassification, 
      detectionResult
    );

    // Add classification method metadata
    enhancedClassification.classificationMethod = usedCNN ? 'CNN' : 'AI Vision';
    enhancedClassification.modelInfo = usedCNN ? 
      { type: 'Trained CNN', accuracy: 'High', processingTime: classification.cnnData?.processingTime } :
      { type: model, accuracy: 'Variable', apiCall: true };

    return enhancedClassification;
  } catch (error: any) {
    console.error('Wound classification error:', error);
    throw new Error(`Wound classification failed: ${error.message}`);
  }
}

function normalizeSize(size: string): string {
  const normalized = size?.toLowerCase();
  if (['small', 'medium', 'large'].includes(normalized)) {
    return normalized;
  }
  return 'medium'; // default
}

function normalizeExudate(exudate: string): string {
  const normalized = exudate?.toLowerCase();
  if (['none', 'low', 'moderate', 'heavy'].includes(normalized)) {
    return normalized;
  }
  return 'moderate'; // default
}

function enhanceClassificationWithDetection(classification: any, detectionResult: any): any {
  if (!detectionResult.detections || detectionResult.detections.length === 0) {
    return classification;
  }
  
  const primaryWound = detectionResult.detections[0];
  
  return {
    ...classification,
    detection: {
      confidence: primaryWound.confidence,
      boundingBox: primaryWound.boundingBox,
      measurements: primaryWound.measurements,
      scaleCalibrated: primaryWound.scaleCalibrated
    },
    size: categorizeSizeFromMeasurements(primaryWound.measurements),
    preciseMeasurements: primaryWound.measurements,
    detectionMetadata: {
      model: detectionResult.model,
      version: detectionResult.version,
      processingTime: detectionResult.processingTime,
      multipleWounds: detectionResult.detections.length > 1
    }
  };
}

function categorizeSizeFromMeasurements(measurements: any): string {
  const areaMm2 = measurements.areaMm2;
  
  if (areaMm2 < 100) return 'small';
  if (areaMm2 < 500) return 'medium';
  return 'large';
}
