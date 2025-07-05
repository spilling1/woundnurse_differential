import { analyzeWoundImage } from "./openai";
import { analyzeWoundImageWithGemini } from "./gemini";
import { storage } from "../storage";
import { cloudWoundDetectionService } from "./cloudWoundDetection";

export async function classifyWound(imageBase64: string, model: string, mimeType: string = 'image/jpeg'): Promise<any> {
  try {
    // Step 1: Perform cloud-based wound detection first
    const detectionResult = await cloudWoundDetectionService.detectWounds(imageBase64);
    
    // Get agent instructions from database to include in analysis
    const agentInstructions = await storage.getActiveAgentInstructions();
    const instructions = agentInstructions?.content || '';
    
    // Step 2: Enhance AI analysis with detection data
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
    
    let classification;
    
    if (model.startsWith('gemini-')) {
      classification = await analyzeWoundImageWithGemini(imageBase64, model, enhancedInstructions);
    } else {
      classification = await analyzeWoundImage(imageBase64, model, mimeType, enhancedInstructions);
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
