import type { Express } from "express";
import { storage } from "../storage";
import { isAuthenticated } from "../replitAuth";
import { generateCarePlan } from "../services/carePlanGenerator";

export function registerAdminRoutes(app: Express): void {
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
        return res.status(404).json({
          code: "NO_INSTRUCTIONS",
          message: "No agent instructions found"
        });
      }
      
      // Return the structured instructions
      res.json({
        systemPrompts: instructions.systemPrompts,
        carePlanStructure: instructions.carePlanStructure,
        specificWoundCare: instructions.specificWoundCare,
        questionsGuidelines: instructions.questionsGuidelines,
        productRecommendations: instructions.productRecommendations,
        lastModified: instructions.updatedAt,
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
      const { caseId, editedCarePlan, rating, nurseNotes, medicalHelpNeeded } = req.body;
      
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
        carePlanUpdated: editedCarePlan !== undefined,
        medicalHelpNeeded: medicalHelpNeeded || false
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
      const { systemPrompts, carePlanStructure, specificWoundCare, questionsGuidelines, productRecommendations } = req.body;
      
      if (!systemPrompts || !carePlanStructure || !specificWoundCare) {
        return res.status(400).json({
          code: "INVALID_CONTENT",
          message: "System prompts, care plan structure, and specific wound care are required"
        });
      }
      
      const currentInstructions = await storage.getActiveAgentInstructions();
      let result;
      
      if (currentInstructions) {
        // Update existing instructions
        result = await storage.updateAgentInstructions(currentInstructions.id, {
          systemPrompts,
          carePlanStructure,
          specificWoundCare,
          questionsGuidelines,
          productRecommendations
        });
      } else {
        // Create new instructions
        result = await storage.createAgentInstructions({
          systemPrompts,
          carePlanStructure,
          specificWoundCare,
          questionsGuidelines,
          productRecommendations,
          version: 1
        });
      }
      
      res.json({
        success: true,
        message: "Agent instructions updated successfully",
        lastModified: result.updatedAt,
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

  // Nurse re-run evaluation with different wound type
  app.post("/api/nurse-rerun-evaluation", isAuthenticated, async (req, res) => {
    try {
      const { caseId, woundType, contextData, clinicalSummary } = req.body;
      
      if (!caseId) {
        return res.status(400).json({
          code: "MISSING_REQUIRED_DATA", 
          message: "Case ID is required"
        });
      }

      // Get the existing assessment data to use the same image and settings
      const existingAssessment = await storage.getWoundAssessment(caseId);
      if (!existingAssessment) {
        return res.status(404).json({
          code: "ASSESSMENT_NOT_FOUND",
          message: "Assessment not found"
        });
      }

      // Determine classification to use
      let classificationToUse;
      if (woundType) {
        // Nurse is overriding the wound type
        classificationToUse = {
          woundType: woundType,
          confidence: 0.95, // High confidence since nurse specified
          source: 'nurse-override',
          originalClassification: existingAssessment.classification
        };
      } else {
        // Use original AI classification but with updated context
        classificationToUse = typeof existingAssessment.classification === 'string' 
          ? JSON.parse(existingAssessment.classification)
          : existingAssessment.classification;
      }

      // Get current agent instructions to include in generation
      const agentInstructions = await storage.getActiveAgentInstructions();
      const agentInstructionsText = agentInstructions ? 
        `${agentInstructions.systemPrompts}\n\n${agentInstructions.carePlanStructure}\n\n${agentInstructions.specificWoundCare}\n\n${agentInstructions.questionsGuidelines || ''}` : '';

      // Merge existing context with nurse updates
      let existingContext = {};
      try {
        existingContext = typeof existingAssessment.contextData === 'string'
          ? JSON.parse(existingAssessment.contextData)
          : existingAssessment.contextData || {};
      } catch (e) {
        existingContext = {};
      }

      const mergedContext = {
        ...existingContext,
        ...contextData,
        ...clinicalSummary,
        nurseReview: true,
        woundTypeOverride: woundType ? true : false,
        agentInstructions: agentInstructionsText
      };

      // Generate new care plan with image data
      const carePlan = await generateCarePlan(
        existingAssessment.audience,
        classificationToUse,
        mergedContext,
        existingAssessment.model,
        existingAssessment.imageData,
        existingAssessment.imageMimeType
      );

      res.json({ 
        success: true, 
        carePlan,
        woundType: woundType || classificationToUse.woundType,
        classification: classificationToUse 
      });

    } catch (error: any) {
      console.error('Nurse re-run evaluation error:', error);
      res.status(500).json({
        code: "RERUN_EVALUATION_ERROR",
        message: error.message || "Failed to re-run evaluation"
      });
    }
  });

  // Add additional instructions to agent guidelines
  app.post("/api/agents/add-instructions", isAuthenticated, async (req, res) => {
    try {
      const { instructions } = req.body;
      
      if (!instructions || !instructions.trim()) {
        return res.status(400).json({
          code: "EMPTY_INSTRUCTIONS",
          message: "Instructions cannot be empty"
        });
      }

      // Get current agent instructions
      const currentInstructions = await storage.getActiveAgentInstructions();
      
      // Append new instructions with timestamp and separator
      const timestamp = new Date().toISOString().split('T')[0];
      const additionalText = `\n\n## Nurse Instructions Added ${timestamp}\n\n${instructions.trim()}`;
      
      let updatedContent;
      if (currentInstructions) {
        updatedContent = currentInstructions.systemPrompts + additionalText;
        await storage.updateAgentInstructions(currentInstructions.id, {
          systemPrompts: updatedContent
        });
      } else {
        updatedContent = `# AI Agent Instructions\n\n## Initial Instructions\n\nProvide comprehensive wound care assessments.${additionalText}`;
        await storage.createAgentInstructions({
          systemPrompts: updatedContent,
          carePlanStructure: 'Default care plan structure',
          specificWoundCare: 'Default wound care instructions',
          isActive: true
        });
      }

      res.json({ 
        success: true, 
        message: "Instructions added successfully",
        updatedContent 
      });

    } catch (error: any) {
      console.error('Add agent instructions error:', error);
      res.status(500).json({
        code: "ADD_INSTRUCTIONS_ERROR",
        message: error.message || "Failed to add instructions"
      });
    }
  });
} 