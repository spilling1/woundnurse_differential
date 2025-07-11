import { analyzeWoundImage } from "./openai";
import { analyzeWoundImageWithGemini } from "./gemini";
import { storage } from "../storage";
import { woundDetectionService } from "./woundDetection";
import { cnnWoundClassifier, convertCNNToStandardClassification } from "./cnnWoundClassifier";
import { whyClassificationLogger } from "./whyClassificationLogger";

// Helper function to validate wound type against configured wound types
async function validateWoundType(detectedWoundType: string): Promise<{
  isValid: boolean;
  validTypes: string[];
  closestMatch?: string;
}> {
  try {
    // Get enabled wound types from database
    const enabledWoundTypes = await storage.getEnabledWoundTypes();
    console.log('ValidateWoundType: Enabled wound types:', enabledWoundTypes.map(t => `${t.displayName} (enabled: ${t.isEnabled})`));
    const validTypes = enabledWoundTypes.map(type => type.displayName);
    
    // Normalize wound type for comparison (case-insensitive, handle variations)
    const normalizedDetected = detectedWoundType.toLowerCase().trim();
    console.log('ValidateWoundType: Checking detected wound type:', detectedWoundType, 'normalized:', normalizedDetected);
    
    // Check for exact matches first
    const exactMatch = enabledWoundTypes.find(type => 
      type.displayName.toLowerCase() === normalizedDetected ||
      type.name.toLowerCase() === normalizedDetected
    );
    console.log('ValidateWoundType: Exact match found:', exactMatch?.displayName || 'none');
    
    if (exactMatch) {
      return { isValid: true, validTypes };
    }
    
    // Check for partial matches or common variations
    const partialMatch = enabledWoundTypes.find(type => {
      const typeName = type.displayName.toLowerCase();
      const typeKey = type.name.toLowerCase();
      
      const containsTypeName = normalizedDetected.includes(typeName);
      const containsTypeKey = normalizedDetected.includes(typeKey);
      const typeNameContainsDetected = typeName.includes(normalizedDetected);
      const typeKeyContainsDetected = typeKey.includes(normalizedDetected);
      
      if (containsTypeName || containsTypeKey || typeNameContainsDetected || typeKeyContainsDetected) {
        console.log('ValidateWoundType: Partial match found:', type.displayName, 'reasons:', {
          containsTypeName, containsTypeKey, typeNameContainsDetected, typeKeyContainsDetected
        });
        return true;
      }
      return false;
    });
    
    if (partialMatch) {
      return { isValid: true, validTypes, closestMatch: partialMatch.displayName };
    }
    
    // Check for database-stored synonyms (both exact and partial matches)
    for (const woundType of enabledWoundTypes) {
      if (woundType.synonyms && woundType.synonyms.length > 0) {
        const synonyms = woundType.synonyms;
        
        // Check for exact matches first
        const exactSynonymMatch = synonyms.some(synonym => {
          const normalizedSynonym = synonym.toLowerCase().trim();
          return normalizedDetected === normalizedSynonym;
        });
        
        if (exactSynonymMatch) {
          console.log('ValidateWoundType: Exact synonym match found:', woundType.displayName);
          return { isValid: true, validTypes, closestMatch: woundType.displayName };
        }
        
        // Check if detected wound type contains any of the synonyms
        const partialSynonymMatch = synonyms.some(synonym => {
          const normalizedSynonym = synonym.toLowerCase().trim();
          const detectedContainsSynonym = normalizedDetected.includes(normalizedSynonym);
          const synonymContainsDetected = normalizedSynonym.includes(normalizedDetected);
          
          // For multi-word synonyms, also check if all words in the synonym are present in the detected type
          const synonymWords = normalizedSynonym.split(' ').filter(word => word.length > 2); // Only check meaningful words
          const allWordsPresent = synonymWords.length > 0 && synonymWords.every(word => 
            normalizedDetected.includes(word)
          );
          
          if (detectedContainsSynonym || synonymContainsDetected || allWordsPresent) {
            console.log('ValidateWoundType: Partial synonym match found:', woundType.displayName, 
              'synonym:', synonym, 'detected:', normalizedDetected, 
              'reasons:', { detectedContainsSynonym, synonymContainsDetected, allWordsPresent });
            return true;
          }
          return false;
        });
        
        if (partialSynonymMatch) {
          return { isValid: true, validTypes, closestMatch: woundType.displayName };
        }
      }
    }
    
    return { isValid: false, validTypes };
  } catch (error) {
    console.error('Error validating wound type:', error);
    // In case of error, be strict and reject the classification
    return { isValid: false, validTypes: ['Error: Could not validate wound type'] };
  }
}

export async function classifyWound(imageBase64: string, model: string, mimeType: string = 'image/jpeg', sessionId?: string, userInfo?: { userId: string; email: string }, bodyRegion?: { id: string; name: string }): Promise<any> {
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
    
    // Step 2: First run AI classification INDEPENDENTLY (no YOLO context)
    if (!usedCNN) {
      console.log('WoundClassifier: Starting independent AI classification...');
      
      // Log body region information if provided
      if (bodyRegion) {
        console.log('WoundClassifier: Body region information received:', {
          id: bodyRegion.id,
          name: bodyRegion.name
        });
      } else {
        console.log('WoundClassifier: No body region information provided');
      }
      
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
      
      // Get comprehensive agent instructions from database (WITHOUT YOLO context)
      const agentInstructions = await storage.getActiveAgentInstructions();
      
      // Get wound-specific instructions from database
      const woundTypes = await storage.getEnabledWoundTypes();
      const woundSpecificInstructions = woundTypes.map(wt => 
        `${wt.displayName}: ${wt.instructions || 'Standard assessment guidelines'}`
      ).join('\n\n');
      
      let instructions = agentInstructions ? 
        `${agentInstructions.systemPrompts}\n\n${agentInstructions.carePlanStructure}\n\n${agentInstructions.specificWoundCare}\n\n${agentInstructions.questionsGuidelines || ''}\n\nWOUND TYPE SPECIFIC INSTRUCTIONS:\n${woundSpecificInstructions}` : '';
      
      // Add enhanced differential diagnosis instructions
      instructions += `\n\nDIFFERENTIAL DIAGNOSIS REQUIREMENTS:
You MUST provide a comprehensive differential diagnosis with:
1. Primary diagnosis with confidence percentage
2. At least 2-3 alternative possibilities with confidence percentages
3. All confidence percentages MUST add up to 100%
4. Detailed clinical reasoning for each possibility
5. Specific anatomical and visual features supporting each diagnosis

ENHANCED DIAGNOSTIC CRITERIA:
- Use anatomical location to guide differential diagnosis
- Consider wound bed characteristics, exudate patterns, and surrounding tissue
- Integrate comprehensive wound care knowledge from specific wound care instructions
- Apply evidence-based diagnostic criteria for each wound type
- Factor in common presentation patterns and risk factors

CONFIDENCE SCORING:
- High confidence (85-95%): Clear diagnostic indicators with minimal ambiguity
- Moderate confidence (70-84%): Strong indicators but some differential possibilities
- Lower confidence (50-69%): Multiple viable possibilities requiring clinical correlation
- All probabilities in differential diagnosis must sum to 100%`;
      
      // Add body region information if provided
      if (bodyRegion) {
        // Handle nested body region structure
        const regionName = bodyRegion.name || bodyRegion.id?.name || bodyRegion.id?.id || 'Unknown location';
        const bodyRegionContext = `\n\nBODY REGION CONTEXT:\nThe wound is located on: ${regionName}\nThis anatomical location may provide important context for:\n- Wound type classification (e.g., diabetic ulcers commonly on feet, pressure ulcers on bony prominences)\n- Risk factors and contributing factors\n- Healing considerations and treatment approach\n- Differential diagnosis considerations\n\nConsider this location information when analyzing the wound characteristics and making your assessment.`;
        instructions += bodyRegionContext;
        console.log('WoundClassifier: Added body region context to AI instructions:', bodyRegionContext);
      }
      
      // Independent AI classification first
      if (model.startsWith('gemini-')) {
        try {
          classification = await analyzeWoundImageWithGemini(imageBase64, model, instructions);
        } catch (geminiError: any) {
          if (geminiError.message?.includes('quota') || 
              geminiError.message?.includes('RESOURCE_EXHAUSTED') || 
              geminiError.message?.includes('overloaded') ||
              geminiError.message?.includes('503') ||
              geminiError.message?.includes('UNAVAILABLE')) {
            console.log('WoundClassifier: Gemini service temporarily unavailable, automatically switching to GPT-4o');
            classification = await analyzeWoundImage(imageBase64, 'gpt-4o', mimeType, instructions);
          } else {
            throw geminiError;
          }
        }
      } else {
        classification = await analyzeWoundImage(imageBase64, model, mimeType, instructions);
      }
      
      console.log(`WoundClassifier: Independent AI classification complete: ${classification.woundType} (${(classification.confidence * 100).toFixed(1)}% confidence)`);
      
      console.log('WoundClassifier: BEFORE differential diagnosis check - about to check structure');
      
      try {
        // ALWAYS ensure differential diagnosis is present - force creation regardless of AI response
        console.log('WoundClassifier: Checking differential diagnosis structure:', classification.differentialDiagnosis);
        const hasDifferentialDiagnosis = classification.differentialDiagnosis && 
                                        classification.differentialDiagnosis.possibleTypes && 
                                        classification.differentialDiagnosis.possibleTypes.length > 1;
        
        console.log('WoundClassifier: Has differential diagnosis?', hasDifferentialDiagnosis);
        
        // FORCE differential diagnosis creation for all assessments to ensure multiple possibilities are shown
        if (!hasDifferentialDiagnosis) {
        console.log('WoundClassifier: No differential diagnosis returned, creating comprehensive fallback...');
        
        // Create comprehensive differential diagnosis based on wound type and location
        const primaryType = classification.woundType;
        const location = classification.location?.toLowerCase() || '';
        const primaryConfidence = classification.confidence || 0.7;
        
        let secondaryTypes = [];
        let tertiary = null;
        
        if (primaryType.toLowerCase().includes('pressure')) {
          if (location.includes('heel') || location.includes('foot')) {
            secondaryTypes = ['Diabetic Ulcer', 'Arterial Ulcer'];
            tertiary = 'Neuropathic Ulcer';
          } else {
            secondaryTypes = ['Skin Breakdown', 'Traumatic Wound'];
            tertiary = 'Venous Ulcer';
          }
        } else if (primaryType.toLowerCase().includes('diabetic')) {
          secondaryTypes = ['Pressure Ulcer', 'Arterial Ulcer'];
          tertiary = 'Infectious Wound';
        } else if (primaryType.toLowerCase().includes('venous')) {
          secondaryTypes = ['Arterial Ulcer', 'Traumatic Wound'];
          tertiary = 'Pressure Ulcer';
        } else {
          secondaryTypes = ['Pressure Ulcer', 'Venous Ulcer'];
          tertiary = 'Traumatic Wound';
        }
        
        const secondaryConfidence = Math.max(0.15, (1 - primaryConfidence) * 0.5);
        const tertiaryConfidence = Math.max(0.15, (1 - primaryConfidence) * 0.3);
        
        // Create comprehensive differential diagnosis
        const possibleTypes = [
          {
            woundType: primaryType,
            confidence: primaryConfidence,
            reasoning: `Primary diagnosis based on visual assessment showing characteristic features`
          },
          {
            woundType: secondaryTypes[0],
            confidence: secondaryConfidence,
            reasoning: `Alternative possibility based on anatomical location and wound appearance`
          }
        ];
        
        // Add third possibility if confidence is not too high
        if (primaryConfidence < 0.85 && tertiary) {
          possibleTypes.push({
            woundType: tertiary,
            confidence: tertiaryConfidence,
            reasoning: `Less likely but possible differential based on clinical presentation`
          });
        }
        
        classification.differentialDiagnosis = {
          possibleTypes: possibleTypes,
          questionsToAsk: [
            'Does the patient have diabetes or a history of high blood sugar?',
            'Is the patient able to move around normally or are they bedridden/wheelchair-bound?',
            'How did this wound first start or what caused it?'
          ]
        };
        
        console.log('WoundClassifier: Added comprehensive fallback differential diagnosis with', possibleTypes.length, 'possibilities');
      }
      
      } catch (differentialError) {
        console.error('WoundClassifier: Error in differential diagnosis creation:', differentialError);
        // Force create minimal differential diagnosis even if there's an error
        classification.differentialDiagnosis = {
          possibleTypes: [
            {
              woundType: classification.woundType,
              confidence: classification.confidence || 0.7,
              reasoning: 'Primary diagnosis based on visual assessment'
            },
            {
              woundType: 'Alternative diagnosis requires additional clinical information',
              confidence: 0.3,
              reasoning: 'Multiple possibilities exist - clinical assessment needed'
            }
          ],
          questionsToAsk: [
            'What medical history does the patient have?',
            'How did this wound begin?',
            'What treatments have been tried?'
          ]
        };
        console.log('WoundClassifier: Created minimal differential diagnosis due to error');
      }
      
      // Step 2.5: Check wound type support (but don't throw error - let frontend handle redirect)
      const validationResult = await validateWoundType(classification.woundType);
      if (!validationResult.isValid) {
        console.log(`WoundClassifier: Invalid wound type detected: ${classification.woundType} - flagging for frontend redirect`);
        
        // Add flags for frontend to handle unsupported wound types
        classification.unsupportedWoundType = true;
        classification.supportedTypes = validationResult.validTypes;
        classification.reasoning = classification.additionalObservations || 'AI visual analysis and pattern recognition';
        
        // Log the invalid wound type attempt
        try {
          await storage.createAiInteraction({
            caseId: sessionId || 'temp-session',
            stepType: 'wound_type_validation',
            modelUsed: model,
            promptSent: `Validating wound type: ${classification.woundType}`,
            responseReceived: JSON.stringify(validationResult),
            parsedResult: validationResult,
            errorOccurred: true,
            errorMessage: `Invalid wound type: ${classification.woundType}. Configured types: ${validationResult.validTypes.join(', ')}`
          });
        } catch (logError) {
          console.error('WoundClassifier: Error logging validation failure:', logError);
        }
      } else {
        console.log(`WoundClassifier: Wound type validation passed: ${classification.woundType}`);
      }
      
      // Log the independent classification
      try {
        console.log(`WoundClassifier: Attempting to log independent classification for case: ${sessionId}`);
        await storage.createAiInteraction({
          caseId: sessionId || 'temp-session',
          stepType: 'independent_classification',
          modelUsed: model,
          promptSent: instructions,
          responseReceived: JSON.stringify(classification),
          parsedResult: classification,
          confidenceScore: Math.round(classification.confidence * 100),
          errorOccurred: false,
        });
        console.log(`WoundClassifier: Successfully logged independent classification for case: ${sessionId}`);
      } catch (logError) {
        console.error('WoundClassifier: Error logging AI interaction:', logError);
      }
    }
    
    // Step 2.8: ALWAYS ensure differential diagnosis is present with normalized probabilities
    console.log('WoundClassifier: BEFORE differential diagnosis check - about to check structure');
    
    try {
      // Import differential diagnosis service for probability normalization
      const { differentialDiagnosisService } = await import('./differentialDiagnosisService');
      
      // ALWAYS ensure differential diagnosis is present - force creation regardless of AI response
      console.log('WoundClassifier: Checking differential diagnosis structure:', classification.differentialDiagnosis);
      const hasDifferentialDiagnosis = classification.differentialDiagnosis && 
                                      classification.differentialDiagnosis.possibleTypes && 
                                      classification.differentialDiagnosis.possibleTypes.length > 1;
      
      console.log('WoundClassifier: Has differential diagnosis?', hasDifferentialDiagnosis);
      
      // FORCE differential diagnosis creation for all assessments to ensure multiple possibilities are shown
      if (!hasDifferentialDiagnosis) {
        console.log('WoundClassifier: No differential diagnosis returned, creating comprehensive fallback...');
        
        // Create comprehensive differential diagnosis based on wound type and location
        const primaryType = classification.woundType;
        const location = classification.location?.toLowerCase() || '';
        const primaryConfidence = classification.confidence || 0.7;
        
        let secondaryTypes = [];
        let tertiary = null;
        
        if (primaryType.toLowerCase().includes('pressure')) {
          if (location.includes('heel') || location.includes('foot')) {
            secondaryTypes = ['Diabetic Ulcer', 'Arterial Ulcer'];
            tertiary = 'Neuropathic Ulcer';
          } else {
            secondaryTypes = ['Skin Breakdown', 'Traumatic Wound'];
            tertiary = 'Venous Ulcer';
          }
        } else if (primaryType.toLowerCase().includes('diabetic')) {
          secondaryTypes = ['Pressure Ulcer', 'Arterial Ulcer'];
          tertiary = 'Infectious Wound';
        } else if (primaryType.toLowerCase().includes('venous')) {
          secondaryTypes = ['Arterial Ulcer', 'Traumatic Wound'];
          tertiary = 'Pressure Ulcer';
        } else {
          secondaryTypes = ['Pressure Ulcer', 'Venous Ulcer'];
          tertiary = 'Traumatic Wound';
        }
        
        const secondaryConfidence = Math.max(0.15, (1 - primaryConfidence) * 0.5);
        const tertiaryConfidence = Math.max(0.1, (1 - primaryConfidence) * 0.3);
        
        // Create comprehensive differential diagnosis
        let possibleTypes = [
          {
            woundType: primaryType,
            confidence: primaryConfidence,
            reasoning: `Primary diagnosis based on visual assessment showing characteristic features`
          },
          {
            woundType: secondaryTypes[0],
            confidence: secondaryConfidence,
            reasoning: `Alternative possibility based on anatomical location and wound appearance`
          }
        ];
        
        // Add third possibility if confidence is not too high
        if (primaryConfidence < 0.85 && tertiary) {
          possibleTypes.push({
            woundType: tertiary,
            confidence: tertiaryConfidence,
            reasoning: `Less likely but possible differential based on clinical presentation`
          });
        }
        
        // NORMALIZE PROBABILITIES TO ADD UP TO 100%
        possibleTypes = differentialDiagnosisService.normalizeProbabilities(possibleTypes);
        
        classification.differentialDiagnosis = {
          possibleTypes: possibleTypes,
          questionsToAsk: [
            'Does the patient have diabetes or a history of high blood sugar?',
            'Is the patient able to move around normally or are they bedridden/wheelchair-bound?',
            'How did this wound first start or what caused it?'
          ]
        };
        
        console.log('WoundClassifier: Added comprehensive fallback differential diagnosis with', possibleTypes.length, 'possibilities');
      } else {
        console.log('WoundClassifier: Differential diagnosis already present with', classification.differentialDiagnosis.possibleTypes.length, 'possibilities');
        
        // Filter out possibilities with confidence 15% or below BEFORE normalization
        const filteredPossibilities = classification.differentialDiagnosis.possibleTypes.filter(type => 
          type.confidence > 0.15
        );
        
        console.log(`WoundClassifier: Filtered out low-confidence possibilities, ${classification.differentialDiagnosis.possibleTypes.length - filteredPossibilities.length} removed`);
        
        // If we have at least 2 possibilities after filtering, use them
        if (filteredPossibilities.length >= 2) {
          classification.differentialDiagnosis.possibleTypes = filteredPossibilities;
        } else {
          // Keep at least 2 possibilities even if below 15%
          classification.differentialDiagnosis.possibleTypes = classification.differentialDiagnosis.possibleTypes.slice(0, 2);
        }
        
        // NORMALIZE EXISTING DIFFERENTIAL DIAGNOSIS PROBABILITIES AFTER FILTERING
        if (classification.differentialDiagnosis.possibleTypes) {
          classification.differentialDiagnosis.possibleTypes = differentialDiagnosisService.normalizeProbabilities(
            classification.differentialDiagnosis.possibleTypes
          );
        }
        console.log('WoundClassifier: Filtered and normalized differential diagnosis with', classification.differentialDiagnosis.possibleTypes.length, 'possibilities');
      }
      
    } catch (differentialError) {
      console.error('WoundClassifier: Error in differential diagnosis creation:', differentialError);
      // Force create minimal differential diagnosis even if there's an error
      classification.differentialDiagnosis = {
        possibleTypes: [
          {
            woundType: classification.woundType,
            confidence: 0.7,
            reasoning: 'Primary diagnosis based on visual assessment'
          },
          {
            woundType: 'Alternative diagnosis requires additional clinical information',
            confidence: 0.3,
            reasoning: 'Multiple possibilities exist - clinical assessment needed'
          }
        ],
        questionsToAsk: [
          'What medical history does the patient have?',
          'How did this wound begin?',
          'What treatments have been tried?'
        ]
      };
      console.log('WoundClassifier: Created minimal differential diagnosis due to error');
    }
    
    // Step 3: Check if YOLO is enabled, then run detection
    const enabledModels = await storage.getEnabledDetectionModels();
    const yoloEnabled = enabledModels.some(model => model.modelType === 'yolo' && model.isEnabled);
    
    console.log('WoundClassifier: YOLO enabled status:', yoloEnabled);
    
    let detectionResult = null;
    if (yoloEnabled) {
      console.log('WoundClassifier: Starting YOLO detection...');
      detectionResult = await woundDetectionService.detectWounds(imageBase64, mimeType);
      console.log('WoundClassifier: YOLO detection complete. Detections found:', detectionResult.detections?.length || 0);
      console.log('WoundClassifier: Detection result model:', detectionResult.model);
    } else {
      console.log('WoundClassifier: YOLO detection disabled, skipping...');
    }
    
    // Step 4: Store the independent AI classification for transparency
    const independentClassification = { ...classification };
    
    // Step 5: If YOLO is enabled AND found something, ask AI to reconsider with YOLO context
    if (yoloEnabled && detectionResult && detectionResult.detections && detectionResult.detections.length > 0) {
      const primaryWound = detectionResult.detections[0];
      const yoloConfidence = (primaryWound.confidence * 100).toFixed(1);
      
      console.log(`WoundClassifier: YOLO detected wound with ${yoloConfidence}% confidence. Asking AI to reconsider...`);
      
      // Map YOLO wound class to readable name
      const woundTypeMapping = {
        'neuropathic_ulcer': 'Neuropathic Ulcer',
        'diabetic_ulcer': 'Diabetic Ulcer',
        'pressure_ulcer': 'Pressure Ulcer',
        'venous_ulcer': 'Venous Ulcer',
        'surgical_wound': 'Surgical Wound'
      };
      
      const yoloWoundType = woundTypeMapping[primaryWound.wound_class] || primaryWound.wound_class;
      
      // Ask AI to reconsider with YOLO context
      const reconsiderPrompt = `You previously classified this wound as: ${independentClassification.woundType} with ${(independentClassification.confidence * 100).toFixed(1)}% confidence.

I now have additional information from a YOLO detection model that was trained on wound images:
- YOLO confidence: ${yoloConfidence}% that this is a ${yoloWoundType}
- YOLO measurements: ${primaryWound.measurements.lengthMm.toFixed(1)}mm x ${primaryWound.measurements.widthMm.toFixed(1)}mm
- YOLO area: ${primaryWound.measurements.areaMm2.toFixed(1)}mm²

IMPORTANT: You must explain your reasoning process clearly. Start your response with something like:
"Given my initial assessment of ${independentClassification.woundType} at ${(independentClassification.confidence * 100).toFixed(1)}% confidence, and considering the YOLO model's ${yoloConfidence}% confidence that this is a ${yoloWoundType}, I need to reconsider..."

Then explain whether you:
- Maintain your original assessment (if YOLO confidence is low or contradicts strong visual evidence)
- Adjust your confidence based on YOLO findings
- Change your classification (if YOLO provides compelling evidence)
- Recommend additional steps (if there's significant disagreement)

Please reconsider your original assessment in light of this additional information. Follow this decision framework:

**If YOLO has high confidence (≥85%):**
- This creates a significant discrepancy that requires critical evaluation
- Consider if this represents a potential misclassification in your visual assessment
- Evaluate whether the YOLO model's training data might be more specialized for this wound type
- If there's substantial disagreement, recommend additional steps like:
  - Requesting more photographs from different angles
  - Suggesting expert medical review
  - Recommending specialized wound care consultation

**If YOLO has moderate confidence (40-84%):**
- Use YOLO findings to refine your assessment
- The model provides helpful supplementary information
- Adjust your confidence based on agreement/disagreement with YOLO findings

**If YOLO has low confidence (<40%):**
- YOLO findings are less reliable
- Maintain primary reliance on your visual assessment
- Use YOLO measurements for sizing information only

**For all cases:**
- Explain your reasoning process clearly
- State whether your confidence increased, decreased, or remained the same
- If recommending additional steps, be specific about what information would be most helpful

Provide your updated assessment in the same JSON format, considering both your visual analysis and the YOLO detection results.`;

      // Get updated classification with YOLO context
      if (model.startsWith('gemini-')) {
        try {
          classification = await analyzeWoundImageWithGemini(imageBase64, model, reconsiderPrompt);
        } catch (geminiError: any) {
          if (geminiError.message?.includes('quota') || geminiError.message?.includes('RESOURCE_EXHAUSTED')) {
            console.log('WoundClassifier: Gemini service temporarily unavailable for reconsideration, keeping original classification');
            // Keep the original classification if Gemini fails
          } else {
            throw geminiError;
          }
        }
      } else {
        classification = await analyzeWoundImage(imageBase64, model, mimeType, reconsiderPrompt);
      }
      
      console.log(`WoundClassifier: AI reconsideration complete: ${classification.woundType} (${(classification.confidence * 100).toFixed(1)}% confidence)`);
      
      // Log the YOLO reconsideration
      try {
        console.log(`WoundClassifier: Attempting to log YOLO reconsideration for case: ${sessionId}`);
        await storage.createAiInteraction({
          caseId: sessionId || 'temp-session',
          stepType: 'yolo_reconsideration',
          modelUsed: model,
          promptSent: reconsiderPrompt,
          responseReceived: JSON.stringify(classification),
          parsedResult: classification,
          confidenceScore: Math.round(classification.confidence * 100),
          errorOccurred: false,
        });
        console.log(`WoundClassifier: Successfully logged YOLO reconsideration for case: ${sessionId}`);
      } catch (logError) {
        console.error('WoundClassifier: Error logging YOLO reconsideration:', logError);
      }
    }
    
    // Store both classifications for transparency
    classification.independentClassification = independentClassification;
    
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
      confidence: classification.confidence || 0.4,  // Lower default to indicate uncertainty when AI doesn't provide confidence
      // CRITICAL: Preserve differential diagnosis data that was created by AI
      differentialDiagnosis: classification.differentialDiagnosis,
      // Preserve any flags that were set during validation
      unsupportedWoundType: classification.unsupportedWoundType,
      supportedTypes: classification.supportedTypes
    };

    // Step 4: Enhance classification with detection data (only if YOLO is enabled)
    let enhancedClassification = normalizedClassification;
    if (yoloEnabled && detectionResult) {
      console.log('WoundClassifier: Enhancing classification with YOLO detection data...');
      enhancedClassification = enhanceClassificationWithDetection(
        normalizedClassification, 
        detectionResult
      );
      console.log('WoundClassifier: Enhanced classification has detection data:', !!enhancedClassification.detection);
      console.log('WoundClassifier: Enhanced classification has detectionMetadata:', !!enhancedClassification.detectionMetadata);
    } else {
      console.log('WoundClassifier: YOLO disabled, skipping detection enhancement...');
    }

    // Add classification method metadata
    enhancedClassification.classificationMethod = usedCNN ? 'CNN' : 'AI Vision';
    enhancedClassification.modelInfo = usedCNN ? 
      { type: 'Trained CNN', accuracy: 'High', processingTime: classification.cnnData?.processingTime } :
      { type: model, accuracy: 'Variable', apiCall: true };

    // Log the classification reasoning if user info is available
    if (userInfo && sessionId) {
      try {
        // Get the AI response for reasoning extraction
        const aiResponse = JSON.stringify(classification);
        
        await whyClassificationLogger.logClassification({
          caseId: sessionId,
          userId: userInfo.userId,
          email: userInfo.email,
          woundType: enhancedClassification.woundType,
          confidence: enhancedClassification.confidence,
          aiModel: model,
          aiResponse,
          detectionMethod: enhancedClassification.classificationMethod,
          bodyRegion: bodyRegion,
          yoloData: yoloEnabled ? {
            enabled: true,
            detectionFound: detectionResult && detectionResult.detections && detectionResult.detections.length > 0,
            yoloConfidence: detectionResult?.detections?.[0]?.confidence,
            originalConfidence: independentClassification?.confidence
          } : { enabled: false, detectionFound: false },
          independentClassification: independentClassification ? {
            woundType: independentClassification.woundType,
            confidence: independentClassification.confidence
          } : undefined
        });
      } catch (logError) {
        console.error('Error logging classification reasoning:', logError);
      }
    }

    console.log('WoundClassifier: Returning classification with keys:', Object.keys(enhancedClassification));
    console.log('WoundClassifier: Classification unsupportedWoundType:', enhancedClassification.unsupportedWoundType);
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
    },
    // Preserve any flags that were set (like unsupportedWoundType)
    unsupportedWoundType: classification.unsupportedWoundType,
    supportedTypes: classification.supportedTypes
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
    
    // AI has already considered YOLO findings in its reconsideration, so don't override
    console.log(`WoundClassifier: YOLO detected ${primaryWound.wound_class} - AI has already considered this in its final decision`);
    
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
    
    // AI has already adjusted confidence based on YOLO findings during reconsideration
    console.log(`WoundClassifier: AI confidence after YOLO reconsideration: ${enhancedClassification.confidence}`);
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
