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
import { logToAgents } from "./services/agentsLogger";
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
      
      // Store assessment
      const assessment = await storage.createWoundAssessment({
        caseId,
        audience,
        model,
        classification,
        carePlan,
        version: "v1.0.0"
      });
      
      // Log to Agents.md
      await logToAgents(assessment, classification, carePlan);
      
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
      
      // Update Agents.md with feedback
      await logToAgents(assessment, assessment.classification, assessment.carePlan, feedback);
      
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

  // Get Agents.md content
  app.get("/api/agents", async (req, res) => {
    try {
      const agentsPath = path.resolve('./Agents.md');
      
      if (!fs.existsSync(agentsPath)) {
        // Create empty file if it doesn't exist
        fs.writeFileSync(agentsPath, '# AI Agent Rules\n\nThis file contains the rules and case history for the AI wound care agent.\n\n');
      }
      
      const content = fs.readFileSync(agentsPath, 'utf8');
      
      res.json({
        content,
        lastModified: fs.statSync(agentsPath).mtime,
        size: content.length
      });
      
    } catch (error: any) {
      console.error('Error reading Agents.md:', error);
      res.status(500).json({
        code: "AGENTS_READ_ERROR",
        message: error.message || "Failed to read Agents.md"
      });
    }
  });

  // Update Agents.md content
  app.post("/api/agents", async (req, res) => {
    try {
      const { content } = req.body;
      
      if (!content || typeof content !== 'string') {
        return res.status(400).json({
          code: "INVALID_CONTENT",
          message: "Content is required and must be a string"
        });
      }
      
      const agentsPath = path.resolve('./Agents.md');
      
      // Create backup before updating
      if (fs.existsSync(agentsPath)) {
        const backupPath = `${agentsPath}.backup.${Date.now()}`;
        fs.copyFileSync(agentsPath, backupPath);
      }
      
      // Write new content
      fs.writeFileSync(agentsPath, content, 'utf8');
      
      res.json({
        success: true,
        message: "Agents.md updated successfully",
        lastModified: fs.statSync(agentsPath).mtime,
        size: content.length
      });
      
    } catch (error: any) {
      console.error('Error updating Agents.md:', error);
      res.status(500).json({
        code: "AGENTS_UPDATE_ERROR",
        message: error.message || "Failed to update Agents.md"
      });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}


