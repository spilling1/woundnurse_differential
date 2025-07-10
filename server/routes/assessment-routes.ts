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
      const { audience, model, ...contextData } = uploadRequestSchema.parse(req.body);
      
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
      
      // Classify wound using AI with sessionId for proper logging
      const classification = await classifyWound(imageBase64, model, req.file.mimetype, sessionId);
      const questionAnalysis = await analyzeAssessmentForQuestions(
        sessionId,
        {
          imageAnalysis: classification,
          audience,
          model,
          previousQuestions: [],
          round: 1,
          instructions: null
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
        contextData, 
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

      const { audience, model } = req.body;
      
      if (!audience || !model) {
        return res.status(400).json({
          code: "MISSING_PARAMS",
          message: "Audience and model are required"
        });
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
        
        classification = await classifyWound(imageBase64, model, primaryImage.mimetype, caseId, userInfo);
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
        
        const singleImageClassification = await classifyWound(primaryImage.base64, model, primaryImage.mimeType, caseId, userInfo);
        
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
          imageCount: files.length
        }
      );

      // Log the detection data to verify it's being passed through
      console.log('API Response - Detection Data:', {
        hasDetection: !!classification.detection,
        hasDetectionMetadata: !!classification.detectionMetadata,
        detectionCount: classification.detectionMetadata?.detectionCount || 0
      });

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
        
        // Return structured error for frontend to handle gracefully
        return res.status(400).json({
          code: "INVALID_WOUND_TYPE",
          message: `The detected wound type "${woundType}" is not currently supported by our system.`,
          woundType,
          confidence: 85, // Default confidence for unsupported types
          redirect: "/unsupported-wound",
          supportedTypes: [
            "Pressure Injury",
            "Venous Ulcer", 
            "Arterial Insufficiency Ulcer",
            "Diabetic Ulcer",
            "Surgical Wound",
            "Traumatic Wound",
            "Ischemic Wound",
            "Radiation Wound",
            "Infectious Wound"
          ]
        });
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
      const { audience, model, previousQuestions, classification, round } = req.body;
      
      const parsedQuestions = JSON.parse(previousQuestions || '[]');
      const parsedClassification = JSON.parse(classification || '{}');
      
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
        model
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
      
      const { audience, userFeedback } = req.body;
      let { model } = req.body;
      
      // Fallback to default model if undefined/null/empty
      if (!model || model === 'undefined' || model === 'null' || model === '') {
        console.log('Model was invalid:', model, '- using gemini-2.5-pro as fallback');
        model = 'gemini-2.5-pro';
      }
      const questions = JSON.parse(req.body.questions || '[]');
      const classification = JSON.parse(req.body.classification || '{}');
      
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

      // Log user's answers to questions
      if (questions.length > 0) {
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
        isFollowUp
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