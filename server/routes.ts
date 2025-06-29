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
  
  // Upload and analyze wound image
  app.post("/api/upload", upload.single('image'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({
          code: "NO_IMAGE",
          message: "No image file provided"
        });
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
        version: "v1.0.0"
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


