import type { Express } from "express";
import { storage } from "../storage";
import { isAuthenticated, isAdmin } from "../replitAuth";
import { generateCarePlan } from "../services/carePlanGenerator";
import { userUpdateSchema, companyCreateSchema, companyUpdateSchema, insertDetectionModelSchema, insertAiAnalysisModelSchema } from "@shared/schema";
import type { InsertDetectionModel, InsertAiAnalysisModel } from "@shared/schema";

export function registerAdminRoutes(app: Express): void {
  // Debug authentication
  app.get("/api/admin/debug", isAuthenticated, async (req, res) => {
    const user = req.user as any;
    const userId = user?.claims?.sub || user?.id;
    
    if (user && userId) {
      try {
        const dbUser = await storage.getUser(userId);
        res.json({ 
          authenticated: req.isAuthenticated(),
          userId: userId,
          user: user,
          dbUser: dbUser,
          isAdmin: dbUser?.role === 'admin'
        });
      } catch (error) {
        res.json({ 
          authenticated: req.isAuthenticated(),
          userId: userId,
          user: user,
          error: error.message
        });
      }
    } else {
      res.json({ 
        authenticated: req.isAuthenticated(),
        user: user
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

  // ===== ADMIN USER MANAGEMENT ROUTES =====
  
  // Get all users (admin only)
  app.get("/api/admin/users", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const users = await storage.getAllUsers();
      res.json(users);
    } catch (error: any) {
      console.error('Error fetching users:', error);
      res.status(500).json({
        code: "FETCH_USERS_ERROR",
        message: error.message || "Failed to fetch users"
      });
    }
  });

  // Get users by company (admin only)
  app.get("/api/admin/users/company/:companyId", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const companyId = parseInt(req.params.companyId);
      if (isNaN(companyId)) {
        return res.status(400).json({
          code: "INVALID_COMPANY_ID",
          message: "Invalid company ID"
        });
      }

      const users = await storage.getUsersByCompany(companyId);
      res.json(users);
    } catch (error: any) {
      console.error('Error fetching users by company:', error);
      res.status(500).json({
        code: "FETCH_USERS_BY_COMPANY_ERROR",
        message: error.message || "Failed to fetch users by company"
      });
    }
  });

  // Update user (admin only)
  app.put("/api/admin/users/:userId", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const userId = req.params.userId;
      const validation = userUpdateSchema.safeParse(req.body);
      
      if (!validation.success) {
        return res.status(400).json({
          code: "INVALID_USER_DATA",
          message: "Invalid user data",
          errors: validation.error.errors
        });
      }

      const user = await storage.updateUser(userId, validation.data);
      res.json(user);
    } catch (error: any) {
      console.error('Error updating user:', error);
      res.status(500).json({
        code: "UPDATE_USER_ERROR",
        message: error.message || "Failed to update user"
      });
    }
  });

  // Delete user (admin only)
  app.delete("/api/admin/users/:userId", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const userId = req.params.userId;
      const success = await storage.deleteUser(userId);
      
      if (!success) {
        return res.status(404).json({
          code: "USER_NOT_FOUND",
          message: "User not found"
        });
      }

      res.json({ success: true, message: "User deleted successfully" });
    } catch (error: any) {
      console.error('Error deleting user:', error);
      res.status(500).json({
        code: "DELETE_USER_ERROR",
        message: error.message || "Failed to delete user"
      });
    }
  });

  // ===== ADMIN ASSESSMENT MANAGEMENT ROUTES =====

  // Get all wound assessments (admin only)
  app.get("/api/admin/assessments", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const assessments = await storage.getAllWoundAssessments();
      res.json(assessments);
    } catch (error: any) {
      console.error('Error fetching assessments:', error);
      res.status(500).json({
        code: "FETCH_ASSESSMENTS_ERROR",
        message: error.message || "Failed to fetch assessments"
      });
    }
  });

  // Get assessments by user (admin only)
  app.get("/api/admin/assessments/user/:userId", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const userId = req.params.userId;
      const assessments = await storage.getWoundAssessmentsByUser(userId);
      res.json(assessments);
    } catch (error: any) {
      console.error('Error fetching assessments by user:', error);
      res.status(500).json({
        code: "FETCH_ASSESSMENTS_BY_USER_ERROR",
        message: error.message || "Failed to fetch assessments by user"
      });
    }
  });

  // ===== ADMIN COMPANY MANAGEMENT ROUTES =====

  // Get all companies (admin only)
  app.get("/api/admin/companies", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const companies = await storage.getAllCompanies();
      res.json(companies);
    } catch (error: any) {
      console.error('Error fetching companies:', error);
      res.status(500).json({
        code: "FETCH_COMPANIES_ERROR",
        message: error.message || "Failed to fetch companies"
      });
    }
  });

  // Get single company (admin only)
  app.get("/api/admin/companies/:id", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      if (isNaN(id)) {
        return res.status(400).json({
          code: "INVALID_COMPANY_ID",
          message: "Invalid company ID"
        });
      }

      const company = await storage.getCompany(id);
      if (!company) {
        return res.status(404).json({
          code: "COMPANY_NOT_FOUND",
          message: "Company not found"
        });
      }

      res.json(company);
    } catch (error: any) {
      console.error('Error fetching company:', error);
      res.status(500).json({
        code: "FETCH_COMPANY_ERROR",
        message: error.message || "Failed to fetch company"
      });
    }
  });

  // Create company (admin only)
  app.post("/api/admin/companies", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const validation = companyCreateSchema.safeParse(req.body);
      
      if (!validation.success) {
        return res.status(400).json({
          code: "INVALID_COMPANY_DATA",
          message: "Invalid company data",
          errors: validation.error.errors
        });
      }

      const company = await storage.createCompany(validation.data);
      res.json(company);
    } catch (error: any) {
      console.error('Error creating company:', error);
      res.status(500).json({
        code: "CREATE_COMPANY_ERROR",
        message: error.message || "Failed to create company"
      });
    }
  });

  // Update company (admin only)
  app.put("/api/admin/companies/:id", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      if (isNaN(id)) {
        return res.status(400).json({
          code: "INVALID_COMPANY_ID",
          message: "Invalid company ID"
        });
      }

      const validation = companyUpdateSchema.safeParse(req.body);
      
      if (!validation.success) {
        return res.status(400).json({
          code: "INVALID_COMPANY_DATA",
          message: "Invalid company data",
          errors: validation.error.errors
        });
      }

      const company = await storage.updateCompany(id, validation.data);
      res.json(company);
    } catch (error: any) {
      console.error('Error updating company:', error);
      res.status(500).json({
        code: "UPDATE_COMPANY_ERROR",
        message: error.message || "Failed to update company"
      });
    }
  });

  // Delete company (admin only)
  app.delete("/api/admin/companies/:id", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      if (isNaN(id)) {
        return res.status(400).json({
          code: "INVALID_COMPANY_ID",
          message: "Invalid company ID"
        });
      }

      const success = await storage.deleteCompany(id);
      
      if (!success) {
        return res.status(404).json({
          code: "COMPANY_NOT_FOUND",
          message: "Company not found"
        });
      }

      res.json({ success: true, message: "Company deleted successfully" });
    } catch (error: any) {
      console.error('Error deleting company:', error);
      res.status(500).json({
        code: "DELETE_COMPANY_ERROR",
        message: error.message || "Failed to delete company"
      });
    }
  });

  // ===== ADMIN DASHBOARD STATS =====

  // Get admin dashboard stats (admin only)
  app.get("/api/admin/dashboard", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const [users, assessments, companies] = await Promise.all([
        storage.getAllUsers(),
        storage.getAllWoundAssessments(),
        storage.getAllCompanies()
      ]);

      const stats = {
        totalUsers: users.length,
        activeUsers: users.filter(u => u.status === 'active').length,
        totalAssessments: assessments.length,
        totalCompanies: companies.length,
        recentUsers: users.slice(0, 5),
        recentAssessments: assessments.slice(0, 10),
        usersByRole: {
          admin: users.filter(u => u.role === 'admin').length,
          user: users.filter(u => u.role === 'user').length,
          nurse: users.filter(u => u.role === 'nurse').length,
          manager: users.filter(u => u.role === 'manager').length,
        }
      };

      res.json(stats);
    } catch (error: any) {
      console.error('Error fetching dashboard stats:', error);
      res.status(500).json({
        code: "FETCH_DASHBOARD_ERROR",
        message: error.message || "Failed to fetch dashboard stats"
      });
    }
  });

  // ===== DETECTION MODEL MANAGEMENT ROUTES =====

  // Get all detection models (admin only)
  app.get("/api/admin/detection-models", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const models = await storage.getAllDetectionModels();
      res.json(models);
    } catch (error: any) {
      console.error('Error fetching detection models:', error);
      res.status(500).json({
        code: "FETCH_DETECTION_MODELS_ERROR",
        message: error.message || "Failed to fetch detection models"
      });
    }
  });

  // Get enabled detection models (admin only)
  app.get("/api/admin/detection-models/enabled", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const models = await storage.getEnabledDetectionModels();
      res.json(models);
    } catch (error: any) {
      console.error('Error fetching enabled detection models:', error);
      res.status(500).json({
        code: "FETCH_ENABLED_DETECTION_MODELS_ERROR",
        message: error.message || "Failed to fetch enabled detection models"
      });
    }
  });

  // Create detection model (admin only)
  app.post("/api/admin/detection-models", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const validation = insertDetectionModelSchema.safeParse(req.body);
      
      if (!validation.success) {
        return res.status(400).json({
          code: "INVALID_DETECTION_MODEL_DATA",
          message: "Invalid detection model data",
          errors: validation.error.errors
        });
      }

      const model = await storage.createDetectionModel(validation.data);
      res.json(model);
    } catch (error: any) {
      console.error('Error creating detection model:', error);
      res.status(500).json({
        code: "CREATE_DETECTION_MODEL_ERROR",
        message: error.message || "Failed to create detection model"
      });
    }
  });

  // Update detection model (admin only)
  app.patch("/api/admin/detection-models/:id", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      if (isNaN(id)) {
        return res.status(400).json({
          code: "INVALID_MODEL_ID",
          message: "Invalid model ID"
        });
      }

      const model = await storage.updateDetectionModel(id, req.body);
      res.json(model);
    } catch (error: any) {
      console.error('Error updating detection model:', error);
      res.status(500).json({
        code: "UPDATE_DETECTION_MODEL_ERROR",
        message: error.message || "Failed to update detection model"
      });
    }
  });

  // Toggle detection model (admin only)
  app.patch("/api/admin/detection-models/:id/toggle", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      if (isNaN(id)) {
        return res.status(400).json({
          code: "INVALID_MODEL_ID",
          message: "Invalid model ID"
        });
      }

      const { enabled } = req.body;
      if (typeof enabled !== 'boolean') {
        return res.status(400).json({
          code: "INVALID_ENABLED_VALUE",
          message: "Enabled must be a boolean value"
        });
      }

      const model = await storage.toggleDetectionModel(id, enabled);
      res.json(model);
    } catch (error: any) {
      console.error('Error toggling detection model:', error);
      res.status(500).json({
        code: "TOGGLE_DETECTION_MODEL_ERROR",
        message: error.message || "Failed to toggle detection model"
      });
    }
  });

  // Delete detection model (admin only)
  app.delete("/api/admin/detection-models/:id", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      if (isNaN(id)) {
        return res.status(400).json({
          code: "INVALID_MODEL_ID",
          message: "Invalid model ID"
        });
      }

      const success = await storage.deleteDetectionModel(id);
      
      if (!success) {
        return res.status(404).json({
          code: "MODEL_NOT_FOUND",
          message: "Detection model not found"
        });
      }

      res.json({ success: true, message: "Detection model deleted successfully" });
    } catch (error: any) {
      console.error('Error deleting detection model:', error);
      res.status(500).json({
        code: "DELETE_DETECTION_MODEL_ERROR",
        message: error.message || "Failed to delete detection model"
      });
    }
  });

  // ===== AI ANALYSIS MODEL MANAGEMENT ROUTES =====

  // Get all AI analysis models (admin only)
  app.get("/api/admin/ai-analysis-models", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const models = await storage.getAllAiAnalysisModels();
      res.json(models);
    } catch (error: any) {
      console.error('Error fetching AI analysis models:', error);
      res.status(500).json({
        code: "FETCH_AI_ANALYSIS_MODELS_ERROR",
        message: error.message || "Failed to fetch AI analysis models"
      });
    }
  });

  // Get enabled AI analysis models (admin only)
  app.get("/api/admin/ai-analysis-models/enabled", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const models = await storage.getEnabledAiAnalysisModels();
      res.json(models);
    } catch (error: any) {
      console.error('Error fetching enabled AI analysis models:', error);
      res.status(500).json({
        code: "FETCH_ENABLED_AI_ANALYSIS_MODELS_ERROR",
        message: error.message || "Failed to fetch enabled AI analysis models"
      });
    }
  });

  // Get default AI analysis model (admin only)
  app.get("/api/admin/ai-analysis-models/default", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const model = await storage.getDefaultAiAnalysisModel();
      res.json(model);
    } catch (error: any) {
      console.error('Error fetching default AI analysis model:', error);
      res.status(500).json({
        code: "FETCH_DEFAULT_AI_ANALYSIS_MODEL_ERROR",
        message: error.message || "Failed to fetch default AI analysis model"
      });
    }
  });

  // Create AI analysis model (admin only)
  app.post("/api/admin/ai-analysis-models", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const validation = insertAiAnalysisModelSchema.safeParse(req.body);
      
      if (!validation.success) {
        return res.status(400).json({
          code: "INVALID_AI_ANALYSIS_MODEL_DATA",
          message: "Invalid AI analysis model data",
          errors: validation.error.errors
        });
      }

      const model = await storage.createAiAnalysisModel(validation.data);
      res.json(model);
    } catch (error: any) {
      console.error('Error creating AI analysis model:', error);
      res.status(500).json({
        code: "CREATE_AI_ANALYSIS_MODEL_ERROR",
        message: error.message || "Failed to create AI analysis model"
      });
    }
  });

  // Update AI analysis model (admin only)
  app.patch("/api/admin/ai-analysis-models/:id", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      if (isNaN(id)) {
        return res.status(400).json({
          code: "INVALID_MODEL_ID",
          message: "Invalid model ID"
        });
      }

      const model = await storage.updateAiAnalysisModel(id, req.body);
      res.json(model);
    } catch (error: any) {
      console.error('Error updating AI analysis model:', error);
      res.status(500).json({
        code: "UPDATE_AI_ANALYSIS_MODEL_ERROR",
        message: error.message || "Failed to update AI analysis model"
      });
    }
  });

  // Toggle AI analysis model enabled/disabled (admin only)
  app.patch("/api/admin/ai-analysis-models/:id/toggle", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      if (isNaN(id)) {
        return res.status(400).json({
          code: "INVALID_MODEL_ID",
          message: "Invalid model ID"
        });
      }

      const { enabled } = req.body;
      if (typeof enabled !== 'boolean') {
        return res.status(400).json({
          code: "INVALID_ENABLED_VALUE",
          message: "Enabled must be a boolean value"
        });
      }

      const model = await storage.toggleAiAnalysisModel(id, enabled);
      res.json(model);
    } catch (error: any) {
      console.error('Error toggling AI analysis model:', error);
      res.status(500).json({
        code: "TOGGLE_AI_ANALYSIS_MODEL_ERROR",
        message: error.message || "Failed to toggle AI analysis model"
      });
    }
  });

  // Set default AI analysis model (admin only)
  app.patch("/api/admin/ai-analysis-models/:id/default", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      if (isNaN(id)) {
        return res.status(400).json({
          code: "INVALID_MODEL_ID",
          message: "Invalid model ID"
        });
      }

      const model = await storage.setDefaultAiAnalysisModel(id);
      res.json(model);
    } catch (error: any) {
      console.error('Error setting default AI analysis model:', error);
      res.status(500).json({
        code: "SET_DEFAULT_AI_ANALYSIS_MODEL_ERROR",
        message: error.message || "Failed to set default AI analysis model"
      });
    }
  });

  // Delete AI analysis model (admin only)
  app.delete("/api/admin/ai-analysis-models/:id", isAuthenticated, isAdmin, async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      if (isNaN(id)) {
        return res.status(400).json({
          code: "INVALID_MODEL_ID",
          message: "Invalid model ID"
        });
      }

      const success = await storage.deleteAiAnalysisModel(id);
      
      if (!success) {
        return res.status(404).json({
          code: "MODEL_NOT_FOUND",
          message: "AI analysis model not found"
        });
      }

      res.json({ success: true, message: "AI analysis model deleted successfully" });
    } catch (error: any) {
      console.error('Error deleting AI analysis model:', error);
      res.status(500).json({
        code: "DELETE_AI_ANALYSIS_MODEL_ERROR",
        message: error.message || "Failed to delete AI analysis model"
      });
    }
  });
} 