import { analyzeWoundImage } from "./openai";
import { analyzeWoundImageWithGemini } from "./gemini";
import { storage } from "../storage";
import { woundDetectionService } from "./woundDetection";
import { cnnWoundClassifier, convertCNNToStandardClassification } from "./cnnWoundClassifier";

export async function classifyWound(imageBase64: string, model: string, mimeType: string = 'image/jpeg'): Promise<any> {
  try {
    // Step 1: TEMPORARILY DISABLED CNN due to poor accuracy (hand classified as diabetic ulcer)
    // TODO: Retrain CNN models with better data quality and validation
    let classification;
    let usedCNN = false;
    
    console.log('CNN temporarily disabled due to accuracy issues - using reliable AI vision models');
    
    // Keeping CNN code for future use once retrained:
    /*
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
    */
    
    // Step 2: Perform YOLO-based wound detection for measurements (regardless of classification method)
    console.log('WoundClassifier: Starting YOLO detection...');
    const detectionResult = await woundDetectionService.detectWounds(imageBase64, mimeType);
    console.log('WoundClassifier: YOLO detection complete. Detections found:', detectionResult.detections?.length || 0);
    console.log('WoundClassifier: Detection result model:', detectionResult.model);
    
    // Step 3: Smart fallback logic - If CNN says "background" but YOLO detects wounds, override CNN
    const shouldOverrideCNN = usedCNN && 
      classification.woundType === 'No wound detected' && 
      detectionResult.detections.length > 0 && 
      detectionResult.detections[0].confidence > 0.3;
    
    if (shouldOverrideCNN) {
      console.log(`CNN said no wound, but YOLO detected ${detectionResult.detections.length} wounds. Using AI vision for classification.`);
      usedCNN = false;
    }
    
    // Step 4: If CNN failed or was overridden, use AI vision models as fallback
    if (!usedCNN) {
      // Initialize classification to prevent undefined errors
      classification = {
        woundType: "Unspecified",
        stage: "Not determined", 
        size: "medium",
        woundBed: "Not assessed",
        exudate: "moderate",
        infectionSigns: [],
        location: "Not specified",
        additionalObservations: "",
        confidence: 0.4
      };
      // Get agent instructions from database to include in analysis
      const agentInstructions = await storage.getActiveAgentInstructions();
      const instructions = agentInstructions ? 
        `${agentInstructions.systemPrompts}\n\n${agentInstructions.carePlanStructure}\n\n${agentInstructions.specificWoundCare}\n\n${agentInstructions.questionsGuidelines || ''}` : '';
      
      // Enhance AI analysis with detection data
      const enhancedInstructions = `${instructions}

CRITICAL WOUND DETECTION DATA FROM YOLO ANALYSIS:
- Detection Method: ${detectionResult.model || 'YOLO v8'}
- Processing Time: ${detectionResult.processingTime || 'N/A'}ms
- Image Dimensions: ${detectionResult.imageWidth}x${detectionResult.imageHeight}px
- Number of wounds detected: ${detectionResult.detections.length}
- Primary wound detection confidence: ${detectionResult.detections[0]?.confidence || 0} (0.0-1.0 scale)
${detectionResult.detections.length > 0 ? `
- Wound bounding box: [${detectionResult.detections[0].boundingBox.x}, ${detectionResult.detections[0].boundingBox.y}, ${detectionResult.detections[0].boundingBox.width}, ${detectionResult.detections[0].boundingBox.height}]
- Wound measurements: ${detectionResult.detections[0].measurements.lengthMm.toFixed(1)}mm x ${detectionResult.detections[0].measurements.widthMm.toFixed(1)}mm
- Wound area: ${detectionResult.detections[0].measurements.areaMm2.toFixed(1)}mm²
- Wound perimeter: ${detectionResult.detections[0].perimeter.toFixed(1)}px
- Scale calibrated: ${detectionResult.detections[0].scaleCalibrated ? 'Yes' : 'No'}
- Reference object detected: ${detectionResult.detections[0].referenceObjectDetected ? 'Yes' : 'No'}
${detectionResult.detections.length > 1 ? `- Additional wounds detected: ${detectionResult.detections.length - 1}` : ''}
` : '- No wounds detected by YOLO system'}

IMPORTANT: Use this YOLO detection data to inform your analysis confidence. If YOLO shows high confidence (>0.7) and precise measurements, increase your classification confidence. If YOLO shows low confidence (<0.3) or no detections, be more cautious in your assessment. Always consider both visual analysis and YOLO detection results together.`;
      
      if (model.startsWith('gemini-')) {
        try {
          classification = await analyzeWoundImageWithGemini(imageBase64, model, enhancedInstructions);
        } catch (geminiError: any) {
          // Check if this is a quota error
          if (geminiError.message?.includes('quota') || geminiError.message?.includes('RESOURCE_EXHAUSTED')) {
            console.log('WoundClassifier: Gemini service temporarily unavailable, automatically switching to GPT-4o');
            // Automatically switch to GPT-4o when Gemini service is unavailable
            classification = await analyzeWoundImage(imageBase64, 'gpt-4o', mimeType, enhancedInstructions);
          } else {
            // Re-throw non-quota errors
            throw geminiError;
          }
        }
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
      confidence: classification.confidence || 0.4  // Lower default to indicate uncertainty when AI doesn't provide confidence
    };

    // Step 4: Enhance classification with detection data
    console.log('WoundClassifier: Enhancing classification with detection data...');
    const enhancedClassification = enhanceClassificationWithDetection(
      normalizedClassification, 
      detectionResult
    );
    console.log('WoundClassifier: Enhanced classification has detection data:', !!enhancedClassification.detection);
    console.log('WoundClassifier: Enhanced classification has detectionMetadata:', !!enhancedClassification.detectionMetadata);

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
  console.log('WoundClassifier: Enhancing with detection result:', {
    hasDetections: detectionResult.detections?.length > 0,
    detectionCount: detectionResult.detections?.length || 0,
    model: detectionResult.model,
    processingTime: detectionResult.processingTime
  });
  
  // Always store detection metadata, even if no detections found
  const enhancedClassification = {
    ...classification,
    detectionMetadata: {
      model: detectionResult.model,
      version: detectionResult.version,
      processingTime: detectionResult.processingTime,
      multipleWounds: detectionResult.detections?.length > 1 || false,
      detectionCount: detectionResult.detections?.length || 0,
      methodUsed: detectionResult.method_used || 'unknown'
    }
  };

  // If YOLO found wounds, use them regardless of AI classification
  if (detectionResult.detections && detectionResult.detections.length > 0) {
    const primaryWound = detectionResult.detections[0];
    console.log(`WoundClassifier: YOLO detected ${primaryWound.wound_class} with confidence ${primaryWound.confidence}`);
    
    // Map YOLO wound types to our classification system
    const woundTypeMapping = {
      'neuropathic_ulcer': 'Neuropathic Ulcer',
      'diabetic_ulcer': 'Diabetic Ulcer',
      'pressure_ulcer': 'Pressure Ulcer',
      'venous_ulcer': 'Venous Ulcer',
      'surgical_wound': 'Surgical Wound'
    };
    
    // Store the original YOLO detected type for transparency
    if (primaryWound.wound_class && woundTypeMapping[primaryWound.wound_class]) {
      enhancedClassification.yoloDetectedType = woundTypeMapping[primaryWound.wound_class];
    }
    
    // Override AI classification if YOLO found something
    if (primaryWound.wound_class && woundTypeMapping[primaryWound.wound_class]) {
      enhancedClassification.woundType = woundTypeMapping[primaryWound.wound_class];
      console.log(`WoundClassifier: Overriding AI classification to ${enhancedClassification.woundType} based on YOLO detection`);
    }
  }
  
  // Add detection data only if wounds were found
  if (detectionResult.detections && detectionResult.detections.length > 0) {
    const primaryWound = detectionResult.detections[0];
    
    enhancedClassification.detection = {
      confidence: primaryWound.confidence,
      boundingBox: primaryWound.boundingBox || primaryWound.bbox,
      measurements: primaryWound.measurements,
      scaleCalibrated: primaryWound.scaleCalibrated
    };
    
    enhancedClassification.size = categorizeSizeFromMeasurements(primaryWound.measurements);
    enhancedClassification.preciseMeasurements = primaryWound.measurements;
    enhancedClassification.detectionMetadata.multipleWounds = detectionResult.detections.length > 1;
    
    // Enhance confidence if YOLO found something
    const originalConfidence = enhancedClassification.confidence || 0;
    const yoloBoost = Math.min(primaryWound.confidence * 0.3, 0.2); // Max 20% boost
    enhancedClassification.confidence = Math.min(originalConfidence + yoloBoost, 1.0);
    console.log(`WoundClassifier: Confidence boosted from ${originalConfidence} to ${enhancedClassification.confidence} due to YOLO detection`);
  }
  
  console.log('WoundClassifier: Enhanced classification result:', {
    hasDetection: !!enhancedClassification.detection,
    hasDetectionMetadata: !!enhancedClassification.detectionMetadata,
    detectionCount: enhancedClassification.detectionMetadata?.detectionCount
  });
  
  return enhancedClassification;
}

function categorizeSizeFromMeasurements(measurements: any): string {
  // Handle both field name formats from different services
  const areaMm2 = measurements.area_mm2 || measurements.areaMm2;
  
  if (!areaMm2 || areaMm2 < 100) return 'small';
  if (areaMm2 < 500) return 'medium';
  return 'large';
}
