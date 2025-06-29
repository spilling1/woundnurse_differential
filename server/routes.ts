import type { Express } from "express";
import { createServer, type Server } from "http";
import multer from "multer";
import * as fs from "fs";
import * as path from "path";
import { storage } from "./storage";
import { uploadRequestSchema, feedbackRequestSchema } from "@shared/schema";
import { validateImage } from "./services/imageProcessor";
import { classifyWound } from "./services/woundClassifier";
import { generateCarePlan } from "./services/carePlanGenerator";
import { generateCaseId } from "./services/utils";
import { setupAuth, isAuthenticated } from "./replitAuth";

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'image/jpeg' || file.mimetype === 'image/png') {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only JPG and PNG are allowed.'));
    }
  },
});

export async function registerRoutes(app: Express): Promise<Server> {
  // Auth middleware
  await setupAuth(app);

  // Auth routes
  app.get('/api/auth/user', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const user = await storage.getUser(userId);
      res.json(user);
    } catch (error) {
      console.error("Error fetching user:", error);
      res.status(500).json({ message: "Failed to fetch user" });
    }
  });

  // Get user's wound assessments
  app.get('/api/my-cases', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const assessments = await storage.getUserWoundAssessments(userId);
      res.json(assessments);
    } catch (error) {
      console.error("Error fetching user cases:", error);
      res.status(500).json({ message: "Failed to fetch cases" });
    }
  });
  
  // Upload and analyze wound image
  app.post("/api/upload", upload.single('image'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({
          code: "NO_IMAGE",
          message: "No image file provided"
        });
      }

      // Get user ID if authenticated (optional)
      let userId = null;
      if (req.isAuthenticated && req.isAuthenticated() && req.user) {
        userId = (req.user as any).claims?.sub;
      }

      // Validate request body
      const { audience, model, ...contextData } = uploadRequestSchema.parse(req.body);
      
      // Validate image
      const imageBase64 = await validateImage(req.file);
      
      // Generate case ID
      const caseId = generateCaseId();
      
      // Classify wound using AI
      const classification = await classifyWound(imageBase64, model);
      
      // Generate care plan
      const carePlan = await generateCarePlan(classification, audience, model, contextData);
      
      // Store assessment with image data and context
      const assessment = await storage.createWoundAssessment({
        caseId,
        userId,
        audience,
        model,
        imageData: imageBase64,
        imageMimeType: req.file.mimetype,
        imageSize: req.file.size,
        classification,
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
      
      // Case is now tracked in database - no additional logging needed
      
      res.json({
        caseId: assessment.caseId,
        classification,
        plan: carePlan,
        model,
        version: assessment.version
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
      
      // Feedback is now tracked in database - no additional logging needed
      
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

  // Follow-up assessment endpoint
  app.post("/api/follow-up/:caseId", upload.single('image'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({
          code: "NO_IMAGE",
          message: "No image file provided"
        });
      }

      const { caseId } = req.params;
      const requestData = followUpRequestSchema.parse({
        caseId,
        ...req.body
      });

      // Get the previous assessment history for this case
      const assessmentHistory = await storage.getWoundAssessmentHistory(caseId);
      if (!assessmentHistory.length) {
        return res.status(404).json({
          code: "CASE_NOT_FOUND",
          message: "Original case not found"
        });
      }

      // Get feedback history for context
      const feedbackHistory = await storage.getFeedbacksByCase(caseId);

      // Validate image
      const imageBase64 = await validateImage(req.file);

      // Classify wound using AI
      const classification = await classifyWound(imageBase64, requestData.model);

      // Build context for care plan generation including previous assessments and progress
      const contextForCarePlan = {
        currentAssessment: {
          classification,
          progressNotes: requestData.progressNotes,
          treatmentResponse: requestData.treatmentResponse,
          contextData: requestData
        },
        previousAssessments: assessmentHistory,
        feedbackHistory: feedbackHistory,
        isFollowUp: true
      };

      // Generate updated care plan
      const carePlan = await generateCarePlan(
        requestData.audience,
        classification,
        contextForCarePlan
      );

      // Get user ID if authenticated
      const userId = req.user?.claims?.sub || null;

      // Store the follow-up assessment
      const assessment = await storage.createFollowUpAssessment({
        caseId,
        userId,
        audience: requestData.audience,
        model: requestData.model,
        imageData: imageBase64,
        imageMimeType: req.file.mimetype,
        imageSize: req.file.size,
        classification,
        carePlan,
        woundOrigin: requestData.woundOrigin || null,
        medicalHistory: requestData.medicalHistory || null,
        woundChanges: requestData.woundChanges || null,
        currentCare: requestData.currentCare || null,
        woundPain: requestData.woundPain || null,
        supportAtHome: requestData.supportAtHome || null,
        mobilityStatus: requestData.mobilityStatus || null,
        nutritionStatus: requestData.nutritionStatus || null,
        progressNotes: requestData.progressNotes,
        treatmentResponse: requestData.treatmentResponse,
        contextData: requestData,
        version: `${assessmentHistory[0].versionNumber + 1}`
      });

      res.json({
        caseId: assessment.caseId,
        version: assessment.versionNumber,
        classification,
        plan: carePlan,
        model: requestData.model,
        previousVersions: assessmentHistory.length
      });

    } catch (error: any) {
      console.error('Follow-up assessment error:', error);
      res.status(500).json({
        code: "FOLLOWUP_ERROR",
        message: error.message || "Failed to process follow-up assessment"
      });
    }
  });

  // Get assessment by case ID
  app.get("/api/assessment/:caseId", async (req, res) => {
    try {
      const { caseId } = req.params;
      const assessment = await storage.getWoundAssessment(caseId);
      
      if (!assessment) {
        return res.status(404).json({
          code: "ASSESSMENT_NOT_FOUND",
          message: "Assessment not found"
        });
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
      const userId = req.user.claims.sub;
      
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

  // Get system status
  app.get("/api/status", async (req, res) => {
    res.json({
      status: "operational",
      version: "v1.0.0",
      models: ["gpt-4o", "gpt-3.5", "gpt-3.5-pro", "gemini-2.5-flash", "gemini-2.5-pro"],
      processingQueue: 0
    });
  });

  // Get Agent Instructions
  app.get("/api/agents", async (req, res) => {
    try {
      const instructions = await storage.getActiveAgentInstructions();
      
      if (!instructions) {
        // Create default instructions if none exist
        const defaultContent = `# AI Agent Instructions for Wound Care Assessment

## Core Rules
1. Always prioritize patient safety and recommend consulting healthcare professionals
2. Provide audience-specific language (family, patient, medical)
3. Base recommendations on wound classification and patient context
4. Include clear medical disclaimers

## Assessment Guidelines
- Analyze wound type, stage, size, and location from images
- Consider patient medical history and current care
- Evaluate pain levels and support systems
- Provide step-by-step care instructions

## Care Plan Format
- Start with medical disclaimer
- Provide immediate care steps
- Include monitoring instructions
- Suggest when to seek professional help
- Tailor language to selected audience`;

        const newInstructions = await storage.createAgentInstructions({
          content: defaultContent,
          version: 1
        });
        
        return res.json({
          content: newInstructions.content,
          lastModified: newInstructions.updatedAt,
          size: newInstructions.content.length,
          version: newInstructions.version
        });
      }
      
      res.json({
        content: instructions.content,
        lastModified: instructions.updatedAt,
        size: instructions.content.length,
        version: instructions.version
      });
      
    } catch (error: any) {
      console.error('Error reading agent instructions:', error);
      res.status(500).json({
        code: "AGENTS_READ_ERROR",
        message: error.message || "Failed to read agent instructions"
      });
    }
  });

  // Save nurse evaluation
  app.post("/api/nurse-evaluation", async (req, res) => {
    try {
      const { caseId, editedCarePlan, rating, nurseNotes } = req.body;
      
      if (!caseId) {
        return res.status(400).json({
          code: "INVALID_CASE_ID",
          message: "Case ID is required"
        });
      }

      // For now, just log the evaluation - could be extended to store in database
      console.log('Nurse Evaluation:', {
        caseId,
        rating,
        nurseNotes: nurseNotes?.substring(0, 100) + '...',
        carePlanUpdated: editedCarePlan !== undefined
      });
      
      res.json({
        success: true,
        message: "Nurse evaluation saved successfully"
      });
      
    } catch (error: any) {
      console.error('Error saving nurse evaluation:', error);
      res.status(500).json({
        code: "NURSE_EVALUATION_ERROR",
        message: error.message || "Failed to save nurse evaluation"
      });
    }
  });

  // Update Agent Instructions
  app.post("/api/agents", async (req, res) => {
    try {
      const { content } = req.body;
      
      if (!content || typeof content !== 'string') {
        return res.status(400).json({
          code: "INVALID_CONTENT",
          message: "Content is required and must be a string"
        });
      }
      
      const currentInstructions = await storage.getActiveAgentInstructions();
      let result;
      
      if (currentInstructions) {
        // Update existing instructions
        result = await storage.updateAgentInstructions(currentInstructions.id, content);
      } else {
        // Create new instructions
        result = await storage.createAgentInstructions({
          content,
          version: 1
        });
      }
      
      res.json({
        success: true,
        message: "Agent instructions updated successfully",
        lastModified: result.updatedAt,
        size: content.length,
        version: result.version
      });
      
    } catch (error: any) {
      console.error('Error updating agent instructions:', error);
      res.status(500).json({
        code: "AGENTS_UPDATE_ERROR",
        message: error.message || "Failed to update agent instructions"
      });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}


