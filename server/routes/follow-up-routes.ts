import type { Express } from "express";
import multer from "multer";
import { storage } from "../storage";
import { followUpRequestSchema } from "@shared/schema";
import { validateImage } from "../services/imageProcessor";
import { classifyWound } from "../services/woundClassifier";
import { generateCarePlan } from "../services/carePlanGenerator";
import { isAuthenticated, optionalAuth } from "../customAuth";

// Separate upload configuration for follow-up that accepts multiple file types
const followUpUpload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB
  },
  fileFilter: (req, file, cb) => {
    // Accept images for wound photos
    if (file.fieldname === 'images') {
      if (file.mimetype === 'image/jpeg' || file.mimetype === 'image/png') {
        cb(null, true);
      } else {
        cb(new Error('Invalid image file type. Only JPG and PNG are allowed.'));
      }
    }
    // Accept any file type for additional files (they're temporary)
    else if (file.fieldname === 'additionalFiles') {
      cb(null, true);
    }
    else {
      cb(new Error('Unexpected field name.'));
    }
  },
});

export function registerFollowUpRoutes(app: Express): void {
  // Follow-up assessment endpoint
  app.post("/api/follow-up/:caseId", optionalAuth, followUpUpload.fields([
    { name: 'images', maxCount: 10 },
    { name: 'additionalFiles', maxCount: 10 }
  ]), async (req, res) => {
    try {
      const files = req.files as { [fieldname: string]: Express.Multer.File[] };
      
      if (!files.images || files.images.length === 0) {
        return res.status(400).json({
          code: "NO_IMAGE",
          message: "At least one wound image is required"
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

      // Validate and process primary image (first image for AI analysis)
      const primaryImage = files.images[0];
      const imageBase64 = await validateImage(primaryImage);

      // Process additional files (extract text content if possible)
      let additionalContext = '';
      if (files.additionalFiles && files.additionalFiles.length > 0) {
        // For now, just note that additional files were provided
        const fileNames = files.additionalFiles.map(f => f.originalname).join(', ');
        additionalContext = `Additional files provided: ${fileNames}`;
      }

      // Classify wound using AI (using primary image)
      const classification = await classifyWound(imageBase64, requestData.model, primaryImage.mimetype);

      // Get the original assessment for audience reference and context
      const originalAssessment = assessmentHistory[0]; // First assessment in history

      // Build context for care plan generation including previous assessments and progress
      const originalContext = originalAssessment.contextData as any || {};
      
      const contextForCarePlan = {
        // Include current assessment data
        progressNotes: requestData.progressNotes,
        treatmentResponse: requestData.treatmentResponse,
        
        // Include questionnaire context from original assessment (from contextData JSON)
        woundOrigin: originalContext.woundOrigin,
        medicalHistory: originalContext.medicalHistory,
        woundChanges: requestData.woundChanges || originalContext.woundChanges,
        currentCare: requestData.currentCare || originalContext.currentCare,
        woundPain: requestData.woundPain || originalContext.woundPain,
        supportAtHome: originalContext.supportAtHome,
        mobilityStatus: originalContext.mobilityStatus,
        nutritionStatus: originalContext.nutritionStatus,
        stressLevel: originalContext.stressLevel,
        comorbidities: originalContext.comorbidities,
        age: originalContext.age,
        obesity: originalContext.obesity,
        medications: originalContext.medications,
        alcoholUse: originalContext.alcoholUse,
        smokingStatus: originalContext.smokingStatus,
        frictionShearing: originalContext.frictionShearing,
        knowledgeDeficits: originalContext.knowledgeDeficits,
        woundSite: originalContext.woundSite,
        
        // Follow-up specific context
        currentAssessment: {
          classification,
          progressNotes: requestData.progressNotes,
          treatmentResponse: requestData.treatmentResponse,
        },
        previousAssessments: assessmentHistory,
        feedbackHistory: feedbackHistory,
        isFollowUp: true
      };
      
      // Generate updated care plan using original case's audience
      const carePlan = await generateCarePlan(
        originalAssessment.audience,
        classification,
        contextForCarePlan,
        requestData.model
      );

      // Get user ID if authenticated
      // Get user ID if authenticated (optional for follow-ups)
      let userId = null;
      if ((req as any).customUser?.id) {
        userId = (req as any).customUser.id;
      }

      // Store the follow-up assessment
      const assessment = await storage.createFollowUpAssessment({
        caseId,
        userId,
        audience: originalAssessment.audience,
        model: requestData.model,
        imageData: imageBase64,
        imageMimeType: primaryImage.mimetype,
        imageSize: primaryImage.size,
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
} 