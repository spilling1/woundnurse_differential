import type { Express } from "express";
import multer from "multer";
import { storage } from "../storage";
import { uploadRequestSchema, feedbackRequestSchema } from "@shared/schema";
import { validateImage } from "../services/imageProcessor";
import { classifyWound } from "../services/woundClassifier";
import { generateCarePlan } from "../services/carePlanGenerator";
import { generateCaseId } from "../services/utils";
import { isAuthenticated } from "../replitAuth";
import { analyzeAssessmentForQuestions } from "../services/agentQuestionService";
import { callOpenAI } from "../services/openai";
import { callGemini } from "../services/gemini";

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

export function registerAssessmentRoutes(app: Express): void {
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
      if (req.user && (req.user as any).sub) {
        userId = (req.user as any).sub;
      }

      // Validate request body
      const { audience, model, ...contextData } = uploadRequestSchema.parse(req.body);
      
      // Validate image
      const imageBase64 = await validateImage(req.file);
      
      // Generate case ID
      const caseId = generateCaseId();
      
      // Classify wound using AI
      const classification = await classifyWound(imageBase64, model, req.file.mimetype);
      
      // Analyze if agent needs to ask questions before generating care plan
      const sessionId = caseId; // Use case ID as session ID for question tracking
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
      
      // Generate care plan with detection information
      const carePlan = await generateCarePlan(
        audience, 
        classification, 
        contextData, 
        model,
        undefined, // imageData not needed for non-vision models here
        undefined, // imageMimeType
        classification.detectionMetadata // Pass detection info
      );
      
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
        detectionData: classification.detectionMetadata || null,
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
  app.get("/api/assessment/:caseId", async (req, res) => {
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

  // New Assessment Flow Routes

  // Step 1: Initial image analysis with AI-generated questions
  app.post("/api/assessment/initial-analysis", upload.single('image'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({
          code: "NO_IMAGE",
          message: "Image is required"
        });
      }

      const { audience, model } = req.body;
      
      if (!audience || !model) {
        return res.status(400).json({
          code: "MISSING_PARAMS",
          message: "Audience and model are required"
        });
      }

      // Validate image
      await validateImage(req.file);
      
      // Convert image to base64
      const imageBase64 = req.file.buffer.toString('base64');
      
      // Classify wound
      const classification = await classifyWound(imageBase64, model, req.file.mimetype);
      
      // Generate AI questions based on image analysis
      const questions = await analyzeAssessmentForQuestions(
        'temp-session-' + Date.now(),
        {
          imageAnalysis: classification,
          audience,
          model
        }
      );

      res.json({
        classification,
        questions: questions || []
      });

    } catch (error: any) {
      console.error('Initial analysis error:', error);
      res.status(500).json({
        code: "ANALYSIS_ERROR",
        message: error.message || "Failed to analyze image"
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
      const instructions = agentInstructions?.content || '';
      
      const contextData = {
        previousQuestions: parsedQuestions,
        classification: parsedClassification,
        round: parseInt(round),
        audience,
        model
      };
      
      // Use the agent question service to determine if more questions are needed
      const sessionId = `follow-up-${Date.now()}`;
      const newQuestions = await analyzeAssessmentForQuestions(sessionId, {
        imageAnalysis: parsedClassification,
        audience,
        model,
        previousQuestions: parsedQuestions,
        round: parseInt(round),
        instructions
      });
      
      res.json({
        questions: newQuestions,
        needsMoreQuestions: newQuestions.length > 0,
        round: parseInt(round)
      });
      
    } catch (error: any) {
      console.error('Follow-up questions error:', error);
      res.status(500).json({
        code: "FOLLOWUP_ERROR",
        message: error.message || "Failed to generate follow-up questions"
      });
    }
  });

  // Step 2: Generate preliminary care plan
  app.post("/api/assessment/preliminary-plan", async (req, res) => {
    try {
      const { imageData, audience, model, questions, classification, selectedAlternative, userFeedback } = req.body;
      
      // Use selected alternative if provided, otherwise use main classification
      const finalClassification = selectedAlternative ? 
        { ...classification, woundType: selectedAlternative } : 
        classification;

      // Generate preliminary care plan with proper context
      const contextData = {
        aiQuestions: questions,
        userFeedback,
        classification: finalClassification
      };

      const carePlan = await generateCarePlan(
        audience,
        finalClassification,
        contextData,
        model,
        undefined, // imageData
        undefined, // imageMimeType
        finalClassification.detectionMetadata // Pass detection info
      );

      // Check if user feedback requires clarifying questions
      let needsAdditionalQuestions = false;
      let feedbackQuestions: string[] = [];
      
      if (userFeedback && userFeedback.trim() !== '') {
        try {
          // Analyze feedback to determine if questions are needed
          const feedbackAnalysisPrompt = `
Analyze this user feedback on a wound assessment and determine if clarifying questions are needed:

ORIGINAL CLASSIFICATION:
- Type: ${finalClassification.woundType}
- Location: ${finalClassification.location}
- Stage: ${finalClassification.stage}

USER FEEDBACK:
"${userFeedback}"

Does this feedback indicate:
1. Contradiction with the visual assessment (e.g., wrong body part, wrong wound type)?
2. Important missing information that affects treatment?
3. Specific concerns that need clarification?

If yes to any, generate 2-3 specific questions to clarify the feedback.
If no clarification needed, respond with "NO_QUESTIONS_NEEDED".

Return either "NO_QUESTIONS_NEEDED" or the questions, one per line.
`;

          let feedbackAnalysis;
          if (model.startsWith('gemini-')) {
            feedbackAnalysis = await callGemini(model, feedbackAnalysisPrompt);
          } else {
            const messages = [
              { role: "system", content: "You are a medical AI assistant analyzing user feedback." },
              { role: "user", content: feedbackAnalysisPrompt }
            ];
            feedbackAnalysis = await callOpenAI(model, messages);
          }
          
          if (feedbackAnalysis && !feedbackAnalysis.includes('NO_QUESTIONS_NEEDED')) {
            feedbackQuestions = feedbackAnalysis.split('\n')
              .map(q => q.trim())
              .filter(q => q.length > 0 && q.includes('?'))
              .slice(0, 3);
            needsAdditionalQuestions = feedbackQuestions.length > 0;
          }
        } catch (error) {
          console.error('Error analyzing feedback for questions:', error);
        }
      }

      // Simulate confidence assessment
      const confidence = finalClassification.confidence || 0.8;
      const needsMoreInfo = confidence < 0.75 || needsAdditionalQuestions;

      const preliminaryPlan = {
        assessment: carePlan.split('\n\n')[0], // First paragraph as assessment
        recommendations: carePlan.split('\n\n').slice(1).filter(p => p.trim()),
        confidence,
        needsMoreInfo,
        additionalQuestions: needsMoreInfo ? (
          needsAdditionalQuestions ? feedbackQuestions : [
            "Can you describe any changes in the wound size over the past week?",
            "What is the patient's pain level on a scale of 1-10?",
            "Are there any signs of infection (redness, warmth, pus)?"
          ]
        ) : undefined
      };

      res.json(preliminaryPlan);

    } catch (error: any) {
      console.error('Preliminary plan error:', error);
      res.status(500).json({
        code: "PLAN_ERROR",
        message: error.message || "Failed to generate preliminary plan"
      });
    }
  });

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
      const { audience, model, userFeedback } = req.body;
      const questions = JSON.parse(req.body.questions || '[]');
      const classification = JSON.parse(req.body.classification || '{}');
      const preliminaryPlan = JSON.parse(req.body.preliminaryPlan || '{}');
      
      // Generate case ID
      const caseId = generateCaseId();
      
      // Convert questions to context data format
      const contextData = questions.reduce((acc: any, q: any) => {
        const key = q.category.toLowerCase().replace(/\s+/g, '');
        acc[key] = q.answer;
        return acc;
      }, {});

      // Generate final care plan with detection information
      const carePlan = await generateCarePlan(
        audience,
        classification,
        { ...contextData, userFeedback, preliminaryPlan },
        model,
        undefined, // imageData
        undefined, // imageMimeType
        classification.detectionMetadata // Pass detection info
      );

      // Process image data
      let imageBase64 = '';
      let imageMimeType = 'image/jpeg';
      let imageSize = 0;
      
      if (req.file) {
        await validateImage(req.file);
        imageBase64 = req.file.buffer.toString('base64');
        imageMimeType = req.file.mimetype;
        imageSize = req.file.size;
      }
      
      // Create wound assessment record
      const assessment = await storage.createWoundAssessment({
        caseId,
        userId: (req as any).user?.claims?.sub || null,
        audience,
        model,
        imageData: imageBase64,
        imageMimeType,
        imageSize,
        classification: JSON.stringify(classification),
        contextData: JSON.stringify(contextData),
        carePlan,
        versionNumber: 1,
        isFollowUp: false
      });

      res.json({
        caseId: assessment.caseId,
        success: true
      });

    } catch (error: any) {
      console.error('Final plan error:', error);
      res.status(500).json({
        code: "FINAL_PLAN_ERROR",
        message: error.message || "Failed to generate final care plan"
      });
    }
  });
} 