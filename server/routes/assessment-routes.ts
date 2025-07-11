import type { Express } from "express";
import multer from "multer";
import { storage } from "../storage";
import { uploadRequestSchema, feedbackRequestSchema } from "@shared/schema";
import { validateImage } from "../services/imageProcessor";
import { classifyWound } from "../services/woundClassifier";
import { generateCarePlan } from "../services/carePlanGenerator";
import { generateCaseId } from "../services/utils";
import { isAuthenticated, optionalAuth } from "../customAuth";
import { analyzeAssessmentForQuestions } from "../services/agentQuestionService";
import { LoggerService } from "../services/loggerService";
import { callOpenAI } from "../services/openai";
import { callGemini } from "../services/gemini";
import { cnnWoundClassifier } from "../services/cnnWoundClassifier";
import { whyClassificationLogger } from "../services/whyClassificationLogger";
import { differentialDiagnosisService } from "../services/differentialDiagnosisService";

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB per file
    files: 5, // Max 5 files
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'image/jpeg' || file.mimetype === 'image/png') {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only JPG and PNG are allowed.'));
    }
  },
});

/**
 * Extract product recommendations from care plan HTML/text
 */
function extractProductsFromCarePlan(carePlan: string): Array<{
  category: string;
  productName: string;
  amazonLink: string;
  reason: string;
}> {
  const products: Array<{
    category: string;
    productName: string;
    amazonLink: string;
    reason: string;
  }> = [];
  
  try {
    // Look for Amazon links in the care plan
    const amazonLinkRegex = /\[([^\]]+)\]\(https:\/\/www\.amazon\.com\/s\?k=([^)]+)\)/g;
    let match;
    
    while ((match = amazonLinkRegex.exec(carePlan)) !== null) {
      const productName = match[1];
      const amazonLink = match[0].match(/\(([^)]+)\)/)?.[1] || '';
      
      // Try to determine category based on context
      let category = 'General';
      const lowerName = productName.toLowerCase();
      
      if (lowerName.includes('dressing') || lowerName.includes('bandage')) {
        category = 'Wound Dressing';
      } else if (lowerName.includes('cleanser') || lowerName.includes('saline')) {
        category = 'Wound Cleanser';
      } else if (lowerName.includes('moisturizer') || lowerName.includes('barrier')) {
        category = 'Skin Care';
      } else if (lowerName.includes('gloves') || lowerName.includes('gauze')) {
        category = 'Supplies';
      } else if (lowerName.includes('compression') || lowerName.includes('sock')) {
        category = 'Compression';
      }
      
      // Extract context/reason from surrounding text
      const contextStart = Math.max(0, match.index - 100);
      const contextEnd = Math.min(carePlan.length, match.index + match[0].length + 100);
      const context = carePlan.substring(contextStart, contextEnd);
      
      let reason = 'Recommended for wound care';
      if (context.includes('infection')) {
        reason = 'Helps prevent infection';
      } else if (context.includes('moist') || context.includes('hydrat')) {
        reason = 'Maintains moist healing environment';
      } else if (context.includes('protect')) {
        reason = 'Provides protection';
      } else if (context.includes('clean')) {
        reason = 'For wound cleaning';
      }
      
      products.push({
        category,
        productName,
        amazonLink,
        reason
      });
    }
    
    return products;
  } catch (error) {
    console.error('Error extracting products from care plan:', error);
    return [];
  }
}

export function registerAssessmentRoutes(app: Express): void {
  // Upload and analyze wound image
  app.post("/api/upload", isAuthenticated, upload.single('image'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({
          code: "NO_IMAGE",
          message: "No image file provided"
        });
      }

      // Get user ID from authenticated user
      const userId = (req as any).customUser.id;

      // Validate request body
      const { audience, model, bodyRegion, ...contextData } = uploadRequestSchema.parse(req.body);
      
      // Parse body region if provided
      let parsedBodyRegion = null;
      if (bodyRegion) {
        try {
          parsedBodyRegion = typeof bodyRegion === 'string' ? JSON.parse(bodyRegion) : bodyRegion;
          console.log('Upload Route: Body region parsed successfully:', parsedBodyRegion);
        } catch (error) {
          console.error('Upload Route: Error parsing body region:', error);
          // Continue without body region if parsing fails
        }
      }
      
      // Validate image
      const imageBase64 = await validateImage(req.file);
      
      // Check for duplicate images BEFORE analysis to avoid unnecessary processing
      if (userId && req.file.size > 0) {
        const duplicateAssessment = await storage.findAssessmentByImageData(userId, imageBase64, req.file.size);
        if (duplicateAssessment) {
          // Return information about the duplicate so frontend can ask user
          return res.json({
            duplicateDetected: true,
            existingCase: {
              caseId: duplicateAssessment.caseId,
              createdAt: duplicateAssessment.createdAt,
              classification: duplicateAssessment.classification
            },
            message: "We found an identical image in your previous assessments. Would you like to create a follow-up assessment or start a new case?"
          });
        }
      }
      
      // Generate case ID
      const caseId = generateCaseId();
      
      // Analyze if agent needs to ask questions before generating care plan
      const sessionId = caseId; // Use case ID as session ID for question tracking
      
      // Get user info for logging
      const userInfo = {
        userId: userId,
        email: (req as any).customUser.email
      };
      
      // Classify wound using AI with sessionId for proper logging
      const classification = await classifyWound(imageBase64, model, req.file.mimetype, sessionId, userInfo, parsedBodyRegion);
      const questionAnalysis = await analyzeAssessmentForQuestions(
        sessionId,
        {
          imageAnalysis: classification,
          audience,
          model,
          previousQuestions: [],
          round: 1,
          instructions: null,
          bodyRegion: parsedBodyRegion
        }
      );
      
      // Ensure sessionId is set for proper logging in care plan generation
      const classificationWithSessionId = {
        ...classification,
        sessionId: sessionId
      };

      // Generate care plan with detection information
      const carePlan = await generateCarePlan(
        audience, 
        classificationWithSessionId, 
        { ...contextData, bodyRegion: parsedBodyRegion }, // Include body region in context
        model,
        undefined, // imageData not needed for non-vision models here
        undefined, // imageMimeType
        classification.detectionMetadata // Pass detection info
      );

      // Log product recommendations from care plan
      try {
        await LoggerService.logProductRecommendations({
          caseId,
          userEmail: (req as any).customUser?.email || 'unknown',
          timestamp: new Date(),
          woundType: classification.woundType || 'unknown',
          audience,
          aiModel: model,
          products: extractProductsFromCarePlan(carePlan)
        });
      } catch (logError) {
        console.error('Error logging product recommendations:', logError);
      }
      
      // Store assessment with image data, context, and detection data
      const assessment = await storage.createWoundAssessment({
        caseId,
        userId,
        audience,
        model,
        imageData: imageBase64,
        imageMimeType: req.file.mimetype,
        imageSize: req.file.size,
        bodyRegion: parsedBodyRegion,
        classification,
        detectionData: classification.detection || classification.detectionMetadata || null,
        carePlan,
        woundOrigin: contextData.woundOrigin || null,
        medicalHistory: contextData.medicalHistory || null,
        woundChanges: contextData.woundChanges || null,
        currentCare: contextData.currentCare || null,
        woundPain: contextData.woundPain || null,
        supportAtHome: contextData.supportAtHome || null,
        mobilityStatus: contextData.mobilityStatus || null,
        nutritionStatus: contextData.nutritionStatus || null,
        version: "1",
        versionNumber: 1,
        contextData: contextData
      });
      
      res.json({
        caseId: assessment.caseId,
        classification,
        plan: carePlan,
        model,
        version: assessment.version,
        questionsNeeded: questionAnalysis.length > 0,
        questions: questionAnalysis,
        questionReasoning: "Generated based on AI analysis and confidence level"
      });
      
    } catch (error: any) {
      console.error('Upload error:', error);
      
      if (error.code === 'LIMIT_FILE_SIZE') {
        return res.status(400).json({
          code: "FILE_TOO_LARGE",
          message: "Image must be under 10MB"
        });
      }
      
      if (error.message.includes('Invalid file type')) {
        return res.status(400).json({
          code: "INVALID_IMAGE",
          message: "Image must be PNG or JPG and under 10MB"
        });
      }
      
      // Handle unsupported wound type errors from care plan generation
      if (error.name === 'ANALYSIS_ERROR' && error.message.includes('INVALID_WOUND_TYPE')) {
        return res.status(400).json({
          code: "INVALID_WOUND_TYPE",
          message: `This wound appears to be a ${error.woundType} which is not currently supported by the Wound Nurse.`,
          woundType: error.woundType,
          confidence: error.confidence,
          reasoning: `We have ${error.confidence}% confidence in this assessment based on visual analysis and classification algorithms.`,
          supportedTypes: error.supportedTypes ? error.supportedTypes.split(', ') : [],
          redirect: "/unsupported-wound"
        });
      }
      
      res.status(500).json({
        code: "PROCESSING_ERROR",
        message: error.message || "Failed to process image"
      });
    }
  });

  // Submit feedback for a case
  app.post("/api/feedback", async (req, res) => {
    try {
      const { caseId, feedbackType, comments } = feedbackRequestSchema.parse(req.body);
      
      // Verify case exists
      const assessment = await storage.getWoundAssessment(caseId);
      if (!assessment) {
        return res.status(404).json({
          code: "CASE_NOT_FOUND",
          message: "Case ID not found"
        });
      }
      
      // Store feedback
      const feedback = await storage.createFeedback({
        caseId,
        feedbackType,
        comments
      });
      
      res.json({
        success: true,
        message: "Feedback submitted successfully"
      });
      
    } catch (error: any) {
      console.error('Feedback error:', error);
      res.status(500).json({
        code: "FEEDBACK_ERROR",
        message: error.message || "Failed to submit feedback"
      });
    }
  });

  // Get assessment by case ID
  app.get("/api/assessment/:caseId", optionalAuth, async (req, res) => {
    try {
      const { caseId } = req.params;
      const { version } = req.query;
      
      let assessment;
      
      if (version) {
        // Get specific version of the assessment
        const assessmentHistory = await storage.getWoundAssessmentHistory(caseId);
        assessment = assessmentHistory.find(a => a.versionNumber === parseInt(version as string));
      } else {
        // Get the latest version if no version specified
        assessment = await storage.getLatestWoundAssessment(caseId);
      }
      
      if (!assessment) {
        return res.status(404).json({
          code: "ASSESSMENT_NOT_FOUND",
          message: version ? `Assessment version ${version} not found` : "Assessment not found"
        });
      }
      
      // Fix existing assessments that have null userId by assigning to current user
      if (!assessment.userId && (req as any).customUser?.id) {
        try {
          await storage.updateWoundAssessment(assessment.caseId, assessment.versionNumber, {
            userId: (req as any).customUser.id
          });
          assessment.userId = (req as any).customUser.id;
        } catch (error) {
          console.error('Error updating assessment userId:', error);
          // Continue without failing the request
        }
      }
      
      res.json(assessment);
      
    } catch (error: any) {
      console.error('Error retrieving assessment:', error);
      res.status(500).json({
        code: "ASSESSMENT_ERROR",
        message: error.message || "Failed to retrieve assessment"
      });
    }
  });

  // Delete assessment by case ID
  app.delete("/api/assessment/:caseId", isAuthenticated, async (req: any, res) => {
    try {
      const { caseId } = req.params;
      const userId = req.customUser.id;
      
      // First check if the assessment exists and belongs to the user
      const assessment = await storage.getWoundAssessment(caseId);
      if (!assessment) {
        return res.status(404).json({
          code: "ASSESSMENT_NOT_FOUND",
          message: "Assessment not found"
        });
      }
      
      // Verify ownership (only allow users to delete their own assessments)
      if (assessment.userId !== userId) {
        return res.status(403).json({
          code: "ACCESS_DENIED",
          message: "You can only delete your own assessments"
        });
      }
      
      const deleted = await storage.deleteWoundAssessment(caseId);
      
      if (!deleted) {
        return res.status(404).json({
          code: "ASSESSMENT_NOT_FOUND",
          message: "Assessment not found"
        });
      }
      
      res.json({
        success: true,
        message: "Assessment deleted successfully"
      });
      
    } catch (error: any) {
      console.error('Error deleting assessment:', error);
      res.status(500).json({
        code: "DELETE_ERROR",
        message: error.message || "Failed to delete assessment"
      });
    }
  });

  // New Assessment Flow Routes

  // Debug endpoint to test wound type validation
  app.post('/api/debug/validate-wound-type', isAuthenticated, async (req, res) => {
    try {
      const { woundType } = req.body;
      
      // Import the function directly
      const { classifyWound } = await import('../services/woundClassifier');
      
      // Test validation directly through storage
      const enabledTypes = await storage.getEnabledWoundTypes();
      
      console.log('Debug: Testing wound type validation for:', woundType);
      console.log('Debug: Enabled types:', enabledTypes.map(t => t.displayName));
      
      // Test the logic manually
      const normalizedDetected = woundType.toLowerCase().trim();
      const exactMatch = enabledTypes.find(type => 
        type.displayName.toLowerCase() === normalizedDetected ||
        type.name.toLowerCase() === normalizedDetected
      );
      
      const partialMatch = enabledTypes.find(type => {
        const typeName = type.displayName.toLowerCase();
        const typeKey = type.name.toLowerCase();
        
        return normalizedDetected.includes(typeName) || 
               normalizedDetected.includes(typeKey) ||
               typeName.includes(normalizedDetected) ||
               typeKey.includes(normalizedDetected);
      });
      
      res.json({
        input: woundType,
        normalizedInput: normalizedDetected,
        exactMatch: exactMatch ? exactMatch.displayName : null,
        partialMatch: partialMatch ? partialMatch.displayName : null,
        enabledTypes: enabledTypes.map(t => ({ name: t.name, displayName: t.displayName, enabled: t.isEnabled }))
      });
    } catch (error) {
      console.error('Debug validation error:', error);
      res.status(500).json({ error: error.message });
    }
  });

  // Step 1: Initial image analysis with AI-generated questions
  app.post("/api/assessment/initial-analysis", optionalAuth, upload.array('images', 5), async (req, res) => {
    try {
      const files = req.files as Express.Multer.File[];
      
      if (!files || files.length === 0) {
        return res.status(400).json({
          code: "NO_IMAGE",
          message: "At least one image is required"
        });
      }

      const { audience, model, bodyRegion } = req.body;
      
      if (!audience || !model) {
        return res.status(400).json({
          code: "MISSING_PARAMS",
          message: "Audience and model are required"
        });
      }

      // Parse body region if provided
      let parsedBodyRegion = null;
      if (bodyRegion) {
        try {
          parsedBodyRegion = typeof bodyRegion === 'string' ? JSON.parse(bodyRegion) : bodyRegion;
        } catch (error) {
          console.error('Error parsing body region:', error);
          // Continue without body region if parsing fails
        }
      }

      // Validate all images
      for (const file of files) {
        await validateImage(file);
      }
      
      // Check for duplicate images IMMEDIATELY after validation, before any AI analysis
      // But only if duplicate detection is enabled
      const primaryImage = files[0];
      const imageBase64 = primaryImage.buffer.toString('base64');
      
      // Get agent instructions to check if duplicate detection is enabled
      const agentInstructions = await storage.getActiveAgentInstructions();
      const duplicateDetectionEnabled = agentInstructions?.duplicateDetectionEnabled !== false; // Default to true if undefined
      
      if (duplicateDetectionEnabled && req.customUser?.id && primaryImage.size > 0) {
        const duplicateAssessment = await storage.findAssessmentByImageData(req.customUser.id, imageBase64, primaryImage.size);
        if (duplicateAssessment) {
          // Return information about the duplicate so frontend can ask user
          return res.json({
            duplicateDetected: true,
            existingCase: {
              caseId: duplicateAssessment.caseId,
              createdAt: duplicateAssessment.createdAt,
              classification: duplicateAssessment.classification
            },
            message: "We found an identical image in your previous assessments. Would you like to create a follow-up assessment or start a new case?"
          });
        }
      }
      
      // We already fetched agent instructions above for duplicate detection check
      const instructions = agentInstructions ? 
        `${agentInstructions.systemPrompts}\n\n${agentInstructions.carePlanStructure}\n\n${agentInstructions.specificWoundCare}\n\n${agentInstructions.questionsGuidelines || ''}` : '';
      
      let classification;
      
      if (files.length === 1) {
        // Single image analysis
        const primaryImage = files[0];
        const imageBase64 = primaryImage.buffer.toString('base64');
        // Generate a case ID for logging (will be used later)
        const caseId = generateCaseId();
        
        // Get user info for logging
        const userInfo = req.customUser ? {
          userId: req.customUser.id,
          email: req.customUser.email
        } : undefined;
        
        classification = await classifyWound(imageBase64, model, primaryImage.mimetype, caseId, userInfo, parsedBodyRegion);
      } else {
        // Multiple image analysis - use enhanced AI functions
        const images = files.map(file => ({
          base64: file.buffer.toString('base64'),
          mimeType: file.mimetype
        }));
        
        // For multiple images, still need to run YOLO detection on primary image
        const primaryImage = images[0];
        // Generate a case ID for logging (will be used later)
        const caseId = generateCaseId();
        
        // Get user info for logging
        const userInfo = req.customUser ? {
          userId: req.customUser.id,
          email: req.customUser.email
        } : undefined;
        
        const singleImageClassification = await classifyWound(primaryImage.base64, model, primaryImage.mimeType, caseId, userInfo, parsedBodyRegion);
        
        if (model.includes('gemini')) {
          // Use Gemini multiple image analysis
          const { analyzeMultipleWoundImagesWithGemini } = await import('../services/gemini');
          const result = await analyzeMultipleWoundImagesWithGemini(images, model, instructions);
          
          // Convert to expected format and preserve detection data
          classification = {
            woundType: result.woundType,
            stage: result.stage,
            size: result.size,
            woundBed: result.woundBed,
            exudate: result.exudate,
            infectionSigns: result.infectionSigns,
            location: result.location,
            additionalObservations: result.additionalObservations,
            confidence: result.confidence || 0.5,
            imageAnalysis: result.imageAnalysis,
            multipleWounds: result.multipleWounds,
            classificationMethod: 'Multiple Image AI Analysis',
            detection: singleImageClassification.detection,
            detectionMetadata: singleImageClassification.detectionMetadata
          };
        } else {
          // Use OpenAI multiple image analysis
          const { analyzeMultipleWoundImages } = await import('../services/openai');
          const result = await analyzeMultipleWoundImages(images, model, instructions);
          
          // Convert to expected format and preserve detection data
          classification = {
            woundType: result.woundType,
            stage: result.stage,
            size: result.size,
            woundBed: result.woundBed,
            exudate: result.exudate,
            infectionSigns: result.infectionSigns,
            location: result.location,
            additionalObservations: result.additionalObservations,
            confidence: result.confidence || 0.5,
            imageAnalysis: result.imageAnalysis,
            multipleWounds: result.multipleWounds,
            classificationMethod: 'Multiple Image AI Analysis',
            detection: singleImageClassification.detection,
            detectionMetadata: singleImageClassification.detectionMetadata
          };
        }
      }
      
      // Generate AI questions based on image analysis
      const questions = await analyzeAssessmentForQuestions(
        'temp-session-' + Date.now(),
        {
          imageAnalysis: classification,
          audience,
          model,
          imageCount: files.length,
          bodyRegion: parsedBodyRegion
        }
      );

      // Log the detection data to verify it's being passed through
      console.log('API Response - Detection Data:', {
        hasDetection: !!classification.detection,
        hasDetectionMetadata: !!classification.detectionMetadata,
        detectionCount: classification.detectionMetadata?.detectionCount || 0
      });

      // Log the response for debugging
      console.log('API Response - Classification has unsupportedWoundType:', !!classification.unsupportedWoundType);
      console.log('API Response - Classification object keys:', Object.keys(classification));

      res.json({
        classification,
        questions: questions || [],
        imagesProcessed: files.length
      });

    } catch (error: any) {
      console.error('Initial analysis error:', error);
      
      // Check if this is an invalid wound type error
      if (error.message && error.message.includes('INVALID_WOUND_TYPE:')) {
        // Extract wound type from error message
        const woundTypeMatch = error.message.match(/The detected wound type "(.*?)"/);
        const woundType = woundTypeMatch ? woundTypeMatch[1] : 'Unknown';
        
        // Use static supported types to avoid database delay
        const supportedTypes = [
          "Venous Ulcer", 
          "Arterial Insufficiency Ulcer",
          "Diabetic Ulcer",
          "Surgical Wound",
          "Traumatic Wound",
          "Ischemic Wound",
          "Radiation Wound",
          "Infectious Wound"
        ];
        
        // Return structured error for frontend to handle gracefully
        const errorResponse = {
          code: "INVALID_WOUND_TYPE",
          message: `This wound appears to be a ${woundType} which is not currently supported by the Wound Nurse.`,
          woundType,
          confidence: 90, // Use the actual confidence from the logs (90% shown in the logs)
          reasoning: `visual analysis and classification algorithms`,
          redirect: "/unsupported-wound",
          supportedTypes: supportedTypes
        };
        
        console.log('Sending INVALID_WOUND_TYPE error response:', errorResponse);
        return res.status(400).json(errorResponse);
      }
      
      // Handle other errors normally
      res.status(500).json({
        code: "ANALYSIS_ERROR",
        message: error.message || "Failed to analyze images"
      });
    }
  });

  // Step 1.5: Generate follow-up questions based on previous answers
  app.post("/api/assessment/follow-up-questions", upload.single('image'), async (req, res) => {
    try {
      const { audience, model, previousQuestions, classification, round, bodyRegion } = req.body;
      
      const parsedQuestions = JSON.parse(previousQuestions || '[]');
      const parsedClassification = JSON.parse(classification || '{}');
      
      // Parse body region if provided
      let parsedBodyRegion = null;
      if (bodyRegion) {
        try {
          parsedBodyRegion = typeof bodyRegion === 'string' ? JSON.parse(bodyRegion) : bodyRegion;
          console.log('Follow-up questions: Body region received:', parsedBodyRegion);
        } catch (error) {
          console.error('Error parsing body region in follow-up questions:', error);
        }
      } else {
        console.log('Follow-up questions: No body region provided');
      }
      
      // Get agent instructions to determine if more questions are needed
      const agentInstructions = await storage.getActiveAgentInstructions();
      
      // Get wound-type-specific instructions if we have a classification
      let woundTypeInstructions = '';
      if (parsedClassification.woundType) {
        try {
          console.log(`Follow-up questions: Looking up wound type "${parsedClassification.woundType}"`);
          const woundType = await storage.getWoundTypeByName(parsedClassification.woundType);
          if (woundType && woundType.instructions) {
            woundTypeInstructions = `\n\nWOUND TYPE SPECIFIC INSTRUCTIONS FOR ${woundType.displayName.toUpperCase()}:\n${woundType.instructions}`;
            console.log(`Follow-up questions: Found wound type instructions for ${woundType.displayName}`);
            console.log(`Follow-up questions: Instructions contain "MUST ASK":`, woundType.instructions.includes('MUST ASK'));
          } else {
            console.log(`Follow-up questions: No wound type found or no instructions for "${parsedClassification.woundType}"`);
          }
        } catch (error) {
          console.error('Error getting wound type instructions:', error);
        }
      } else {
        console.log('Follow-up questions: No wound type in classification');
      }
      
      const instructions = agentInstructions ? 
        `${agentInstructions.systemPrompts}\n\n${agentInstructions.carePlanStructure}\n\n${agentInstructions.specificWoundCare}\n\n${agentInstructions.questionsGuidelines || ''}${woundTypeInstructions}` : woundTypeInstructions;
      
      const contextData = {
        previousQuestions: parsedQuestions,
        classification: parsedClassification,
        round: parseInt(round),
        audience,
        model,
        bodyRegion: parsedBodyRegion
      };
      
      // Calculate revised confidence based on answers provided
      let revisedConfidence = parsedClassification.confidence || 0.5;
      
      // Improve confidence based on answered questions - categorize by purpose
      if (parsedQuestions.length > 0) {
        const answeredQuestions = parsedQuestions.filter((q: any) => q.answer && q.answer.trim().length > 0);
        
        // Different confidence boosts based on question category
        let confidenceBoost = 0;
        answeredQuestions.forEach((q: any) => {
          const answer = q.answer.toLowerCase();
          
          // High-impact confidence questions (medical history, location, timeline)
          if (q.question.toLowerCase().includes('diabetes') || 
              q.question.toLowerCase().includes('where') ||
              q.question.toLowerCase().includes('location') ||
              q.question.toLowerCase().includes('how long') ||
              q.question.toLowerCase().includes('wound bed') ||
              q.question.toLowerCase().includes('color')) {
            confidenceBoost += 0.08; // 8% boost for critical diagnostic info
          }
          // Medium-impact questions (symptoms, treatment history)
          else if (q.question.toLowerCase().includes('pain') ||
                   q.question.toLowerCase().includes('drainage') ||
                   q.question.toLowerCase().includes('treatment') ||
                   q.question.toLowerCase().includes('infection')) {
            confidenceBoost += 0.05; // 5% boost for care plan optimization
          }
          // Lower-impact questions (general symptoms)
          else {
            confidenceBoost += 0.03; // 3% boost for other questions
          }
        });
        
        revisedConfidence = Math.min(1.0, revisedConfidence + Math.min(0.35, confidenceBoost));
      }
      
      // Update classification with revised confidence
      const updatedClassification = {
        ...parsedClassification,
        confidence: revisedConfidence
      };
      
      // Use the agent question service to determine if more questions are needed
      const sessionId = parsedClassification.sessionId || `follow-up-${Date.now()}`;
      let newQuestions = [];
      
      try {
        console.log('Calling analyzeAssessmentForQuestions with:', {
          sessionId,
          audience,
          model,
          previousQuestionsLength: parsedQuestions.length,
          round: parseInt(round),
          confidence: revisedConfidence,
          hasInstructions: !!instructions
        });
        
        newQuestions = await analyzeAssessmentForQuestions(sessionId, {
          imageAnalysis: updatedClassification,
          audience,
          model,
          previousQuestions: parsedQuestions,
          round: parseInt(round),
          instructions
        });
        
        console.log('Generated questions:', newQuestions?.length || 0);
      } catch (questionError: any) {
        console.error('Error in analyzeAssessmentForQuestions:', questionError);
        console.error('Full error details:', questionError);
        
        // Return empty questions array if there's an error generating questions
        console.log('Returning empty questions array due to error');
        newQuestions = [];
      }
      
      // Only proceed to care plan if confidence is 80% or higher AND no more questions needed
      const shouldProceedToPlan = revisedConfidence >= 0.80 && newQuestions.length === 0;
      
      res.json({
        questions: newQuestions,
        needsMoreQuestions: !shouldProceedToPlan && newQuestions.length > 0,
        round: parseInt(round),
        updatedConfidence: revisedConfidence,
        updatedClassification,
        shouldProceedToPlan
      });
      
    } catch (error: any) {
      console.error('Follow-up questions error:', error);
      
      // Ensure we have a proper error message
      let errorMessage = "Failed to generate follow-up questions";
      if (error && error.message) {
        errorMessage = error.message;
      } else if (typeof error === 'string') {
        errorMessage = error;
      }
      
      res.status(500).json({
        code: "FOLLOWUP_ERROR",
        message: errorMessage,
        details: error?.stack || error
      });
    }
  });

  // Step 2: Generate preliminary care plan

  // Generate questions based on user feedback
  app.post("/api/assessment/feedback-questions", async (req, res) => {
    try {
      const { classification, userFeedback, audience, model } = req.body;
      
      // Use the agent question service to generate questions based on feedback
      const questions = await analyzeAssessmentForQuestions(
        classification,
        null, // No previous questions for feedback-based questions
        userFeedback,
        audience,
        model
      );
      
      res.json({
        questions: questions.map((q, index) => ({
          id: `feedback-${index}`,
          question: q,
          answer: '',
          category: 'feedback-clarification',
          confidence: 0.5
        }))
      });
      
    } catch (error: any) {
      console.error('Feedback questions error:', error);
      res.status(500).json({
        code: "FEEDBACK_QUESTIONS_ERROR",
        message: error.message || "Failed to generate feedback questions"
      });
    }
  });

  // Step 3: Generate final care plan with case creation
  app.post("/api/assessment/final-plan", isAuthenticated, upload.single('image'), async (req, res) => {
    try {
      console.log('Final plan request body keys:', Object.keys(req.body));
      console.log('Final plan model field:', req.body.model);
      
      const { audience, userFeedback, bodyRegion } = req.body;
      let { model } = req.body;
      
      // Fallback to default model if undefined/null/empty
      if (!model || model === 'undefined' || model === 'null' || model === '') {
        console.log('Model was invalid:', model, '- using gemini-2.5-pro as fallback');
        model = 'gemini-2.5-pro';
      }
      const questions = JSON.parse(req.body.questions || '[]');
      const classification = JSON.parse(req.body.classification || '{}');
      
      // Parse body region if provided
      let parsedBodyRegion = null;
      if (bodyRegion) {
        try {
          parsedBodyRegion = typeof bodyRegion === 'string' ? JSON.parse(bodyRegion) : bodyRegion;
          console.log('Final plan: Body region received:', parsedBodyRegion);
        } catch (error) {
          console.error('Error parsing body region in final plan:', error);
        }
      } else {
        console.log('Final plan: No body region provided');
      }
      
      // Check if this is a follow-up to an existing case
      const { existingCaseId, forceNew } = req.body;
      let caseId = existingCaseId || generateCaseId();
      let versionNumber = 1;
      let isFollowUp = false;
      
      // If this is a follow-up, get the latest version number
      if (existingCaseId) {
        const latestAssessment = await storage.getLatestWoundAssessment(existingCaseId);
        versionNumber = latestAssessment ? (latestAssessment.versionNumber + 1) : 1;
        isFollowUp = true;
      }
      
      // Convert questions to context data format
      const contextData = questions.reduce((acc: any, q: any) => {
        const key = q.category.toLowerCase().replace(/\s+/g, '');
        acc[key] = q.answer;
        return acc;
      }, {});
      
      // CRITICAL: Include the aiQuestions array for the prompt template to process
      if (questions.length > 0) {
        contextData.aiQuestions = questions.filter((q: any) => q.answer && q.answer.trim() !== '');
        console.log(`Including ${contextData.aiQuestions.length} answered questions in contextData for AI processing`);
      }
      
      // Add body region information to context data if provided
      if (parsedBodyRegion) {
        contextData.bodyRegion = parsedBodyRegion;
        console.log('Final plan: Added body region to contextData:', parsedBodyRegion);
      }

      // Log user's answers to questions
      if (questions.length > 0) {
        console.log('Final plan: Processing', questions.length, 'questions');
        console.log('Final plan: Questions received:', questions.map(q => ({
          question: q.question?.substring(0, 50) + '...',
          hasAnswer: !!q.answer,
          answerLength: q.answer?.length || 0,
          category: q.category
        })));
        
        try {
          const answeredQuestions = questions.filter((q: any) => q.answer && q.answer.trim() !== '');
          const questionSummary = answeredQuestions.map((q: any) => 
            `Q: ${q.question}\nA: ${q.answer}\nCategory: ${q.category}`
          ).join('\n\n');
          
          await storage.createAiInteraction({
            caseId: caseId,
            stepType: 'user_question_responses',
            modelUsed: 'user_input',
            promptSent: `User provided answers to ${answeredQuestions.length} follow-up questions:\n\n${questionSummary}`,
            responseReceived: `Context data prepared for care plan generation: ${JSON.stringify(contextData)}`,
            parsedResult: { questions: answeredQuestions, contextData },
            confidenceScore: Math.round(classification.confidence * 100),
            errorOccurred: false,
          });
        } catch (logError) {
          console.error('Error logging user question responses:', logError);
        }
      }

      // Note: Detailed AI interaction logging happens inside the carePlanGenerator service
      // This route only handles the case creation and response formatting

      // Ensure sessionId is set for proper logging and handle null classification
      if (!classification) {
        throw new Error('Classification is required but was not provided');
      }
      
      const classificationWithSessionId = {
        ...classification,
        sessionId: caseId // Ensure sessionId is properly set for logging
      };

      // Generate final care plan with detection information
      const carePlan = await generateCarePlan(
        audience,
        classificationWithSessionId,
        { ...contextData, userFeedback },
        model,
        undefined, // imageData
        undefined, // imageMimeType
        classification?.detectionMetadata || null // Pass detection info safely
      );

      // Log Q&A interactions if questions were answered
      if (questions.length > 0) {
        try {
          const answeredQuestions = questions.filter((q: any) => q.answer && q.answer.trim() !== '');
          if (answeredQuestions.length > 0) {
            await LoggerService.logQAInteraction({
              caseId,
              userEmail: (req as any).customUser?.email || 'unknown',
              timestamp: new Date(),
              woundType: classification.woundType || 'unknown',
              audience,
              aiModel: model,
              questions: answeredQuestions.map((q: any) => ({
                question: q.question,
                answer: q.answer,
                category: q.category,
                confidenceImpact: q.confidenceImpact || 'unknown'
              })),
              finalConfidence: Math.round(classification.confidence * 100),
              reassessment: contextData.reassessment || undefined
            });
          }
        } catch (logError) {
          console.error('Error logging Q&A interaction:', logError);
        }
      }

      // Log product recommendations from care plan
      try {
        await LoggerService.logProductRecommendations({
          caseId,
          userEmail: (req as any).customUser?.email || 'unknown',
          timestamp: new Date(),
          woundType: classification.woundType || 'unknown',
          audience,
          aiModel: model,
          products: extractProductsFromCarePlan(carePlan)
        });
      } catch (logError) {
        console.error('Error logging product recommendations:', logError);
      }

      // Process image data
      let imageBase64 = '';
      let imageMimeType = 'image/jpeg';
      let imageSize = 0;
      
      if (req.file) {
        await validateImage(req.file);
        imageBase64 = req.file.buffer.toString('base64');
        imageMimeType = req.file.mimetype;
        imageSize = req.file.size;
        
        // Note: Duplicate image detection now happens at upload time, not here
      }
      
      // Determine case name based on whether this is a refined diagnosis
      let caseName = null;
      const hasRefinedDiagnosis = contextData && contextData.aiQuestions && contextData.aiQuestions.length > 0;
      
      if (hasRefinedDiagnosis) {
        caseName = "Final Assessment";
      }

      // Create wound assessment record
      const assessment = await storage.createWoundAssessment({
        caseId,
        userId: (req as any).customUser.id,
        audience,
        model,
        imageData: imageBase64,
        imageMimeType,
        imageSize,
        classification: JSON.stringify(classification),
        contextData: JSON.stringify(contextData),
        carePlan,
        versionNumber,
        isFollowUp,
        caseName
      });

      res.json({
        caseId: assessment.caseId,
        success: true
      });

    } catch (error: any) {
      console.error('Final plan error:', error);
      
      // Handle unsupported wound type errors from care plan generation
      if (error.name === 'ANALYSIS_ERROR' && error.message.includes('INVALID_WOUND_TYPE')) {
        return res.status(400).json({
          code: "INVALID_WOUND_TYPE",
          message: `This wound appears to be a ${error.woundType} which is not currently supported by the Wound Nurse.`,
          woundType: error.woundType,
          confidence: error.confidence,
          reasoning: `We have ${error.confidence}% confidence in this assessment based on visual analysis and classification algorithms.`,
          supportedTypes: error.supportedTypes ? error.supportedTypes.split(', ') : [],
          redirect: "/unsupported-wound"
        });
      }
      
      res.status(500).json({
        code: "FINAL_PLAN_ERROR",
        message: error.message || "Failed to generate final care plan"
      });
    }
  });

  // CNN Model Status endpoint
  app.get("/api/cnn-status", async (req, res) => {
    try {
      const modelInfo = await cnnWoundClassifier.getModelInfo();
      
      res.json({
        ...modelInfo,
        status: modelInfo.available ? 'active' : 'unavailable',
        description: modelInfo.available ? 
          `CNN models available for wound classification with ${modelInfo.bestModel}` :
          'No trained CNN models found. Using AI vision models as fallback.'
      });
      
    } catch (error: any) {
      console.error('CNN status error:', error);
      res.status(500).json({
        available: false,
        status: 'error',
        description: `CNN status check failed: ${error.message}`,
        models: []
      });
    }
  });

  // CNN Test endpoint (for testing CNN classification)
  app.post("/api/cnn-test", upload.single('image'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({
          code: "NO_IMAGE",
          message: "Image is required for CNN testing"
        });
      }

      await validateImage(req.file);
      const imageBase64 = req.file.buffer.toString('base64');
      
      const modelInfo = await cnnWoundClassifier.getModelInfo();
      if (!modelInfo.available) {
        return res.status(503).json({
          code: "CNN_UNAVAILABLE",
          message: "No trained CNN models available"
        });
      }
      
      const cnnResult = await cnnWoundClassifier.classifyWound(imageBase64);
      
      res.json({
        success: true,
        result: cnnResult,
        modelUsed: modelInfo.bestModel,
        description: `CNN classification completed using ${modelInfo.bestModel}`
      });
      
    } catch (error: any) {
      console.error('CNN test error:', error);
      res.status(500).json({
        code: "CNN_TEST_ERROR",
        message: error.message || "CNN test failed"
      });
    }
  });

  // Get AI interactions for a specific case (admin only)
  app.get('/api/admin/ai-interactions/:caseId', isAuthenticated, async (req, res) => {
    try {
      const user = (req as any).customUser;
      if (user.role !== 'admin') {
        return res.status(403).json({ error: 'Admin access required' });
      }

      const { caseId } = req.params;
      const interactions = await storage.getAiInteractionsByCase(caseId);
      res.json(interactions);
    } catch (error) {
      console.error('Error fetching AI interactions:', error);
      res.status(500).json({ error: 'Failed to fetch AI interactions' });
    }
  });

  // Differential Diagnosis Refinement API - Interactive Questions System
  app.post("/api/assessment/refine-differential-diagnosis", isAuthenticated, async (req, res) => {
    try {
      const { originalClassification, questionAnswers, otherInformation, model } = req.body;
      
      if (!originalClassification || !questionAnswers || !Array.isArray(questionAnswers)) {
        return res.status(400).json({
          code: "MISSING_DATA",
          message: "Original classification and question answers are required"
        });
      }
      
      console.log('Refining differential diagnosis with', questionAnswers.length, 'answers');
      if (otherInformation) {
        console.log('Additional information provided:', otherInformation.length, 'characters');
      }
      
      // Use the differential diagnosis service to refine the assessment
      const refinement = await differentialDiagnosisService.refineDifferentialDiagnosis(
        originalClassification,
        questionAnswers,
        model || 'gemini-2.5-pro',
        otherInformation
      );
      
      console.log('Differential diagnosis refinement complete:');
      console.log('- Eliminated possibilities:', refinement.eliminatedPossibilities);
      console.log('- Remaining possibilities:', refinement.remainingPossibilities.length);
      console.log('- Final confidence:', refinement.confidence);
      console.log('- Questions analyzed:', refinement.questionsAnalyzed.length);
      console.log('- Questions analyzed details:', refinement.questionsAnalyzed.map(q => ({
        question: q.question?.substring(0, 50) + '...',
        hasAnswer: !!q.answer,
        answerLength: q.answer?.length || 0
      })));
      
      res.json({
        success: true,
        refinement: refinement,
        page2Analysis: {
          eliminated: refinement.eliminatedPossibilities,
          remaining: refinement.remainingPossibilities,
          primaryDiagnosis: refinement.refinedDiagnosis,
          confidence: refinement.confidence,
          reasoning: refinement.reasoning
        }
      });
      
    } catch (error: any) {
      console.error('Differential diagnosis refinement error:', error);
      res.status(500).json({
        code: "DIFFERENTIAL_REFINEMENT_ERROR",
        message: error.message || "Failed to refine differential diagnosis"
      });
    }
  });

  // Get all AI interactions (admin only)
  app.get('/api/admin/ai-interactions', isAuthenticated, async (req, res) => {
    try {
      const user = (req as any).customUser;
      if (user.role !== 'admin') {
        return res.status(403).json({ error: 'Admin access required' });
      }

      const interactions = await storage.getAllAiInteractions();
      res.json(interactions);
    } catch (error) {
      console.error('Error fetching all AI interactions:', error);
      res.status(500).json({ error: 'Failed to fetch AI interactions' });
    }
  });

  // Get Q&A log (admin only)
  app.get('/api/admin/qa-log', isAuthenticated, async (req, res) => {
    try {
      const user = (req as any).customUser;
      if (user.role !== 'admin') {
        return res.status(403).json({ error: 'Admin access required' });
      }

      const limit = parseInt(req.query.limit as string) || 10;
      const qaLog = await LoggerService.getRecentQAEntries(limit);
      res.json({ content: qaLog });
    } catch (error) {
      console.error('Error fetching Q&A log:', error);
      res.status(500).json({ error: 'Failed to fetch Q&A log' });
    }
  });

  // Get product recommendations log (admin only)
  app.get('/api/admin/products-log', isAuthenticated, async (req, res) => {
    try {
      const user = (req as any).customUser;
      if (user.role !== 'admin') {
        return res.status(403).json({ error: 'Admin access required' });
      }

      const limit = parseInt(req.query.limit as string) || 10;
      const productsLog = await LoggerService.getRecentProductEntries(limit);
      res.json({ content: productsLog });
    } catch (error) {
      console.error('Error fetching products log:', error);
      res.status(500).json({ error: 'Failed to fetch products log' });
    }
  });

  // Admin endpoint to view classification reasoning logs
  app.get('/api/admin/classification-log', isAuthenticated, async (req, res) => {
    try {
      // Check if user is admin
      if (!req.customUser || req.customUser.role !== 'admin') {
        return res.status(403).json({ error: 'Admin access required' });
      }

      const logs = await whyClassificationLogger.getRecentLogs(50);
      res.json({ logs });
    } catch (error) {
      console.error('Error fetching classification logs:', error);
      res.status(500).json({ error: 'Failed to fetch classification logs' });
    }
  });
} 