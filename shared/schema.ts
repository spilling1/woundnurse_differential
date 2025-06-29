import { pgTable, text, serial, integer, timestamp, json, boolean } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const woundAssessments = pgTable("wound_assessments", {
  id: serial("id").primaryKey(),
  caseId: text("case_id").notNull().unique(),
  audience: text("audience").notNull(), // 'family' | 'patient' | 'medical'
  model: text("model").notNull(), // 'gpt-4o' | 'gpt-3.5' | 'gpt-3.5-pro' | 'gemini-2.5-flash' | 'gemini-2.5-pro'
  
  // Image storage
  imageData: text("image_data").notNull(), // Base64 encoded image
  imageMimeType: text("image_mime_type").notNull(), // 'image/jpeg' | 'image/png'
  imageSize: integer("image_size").notNull(), // File size in bytes
  
  // Questionnaire answers
  woundOrigin: text("wound_origin"),
  medicalHistory: text("medical_history"),
  woundChanges: text("wound_changes"),
  currentCare: text("current_care"),
  woundPain: text("wound_pain"),
  supportAtHome: text("support_at_home"),
  mobilityStatus: text("mobility_status"),
  nutritionStatus: text("nutrition_status"),
  
  // AI Analysis results
  classification: json("classification"), // wound type, stage, size, etc.
  carePlan: text("care_plan").notNull(),
  
  version: text("version").notNull().default("v1.0.0"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const feedbacks = pgTable("feedbacks", {
  id: serial("id").primaryKey(),
  caseId: text("case_id").notNull(),
  feedbackType: text("feedback_type").notNull(), // 'helpful' | 'not-helpful'
  comments: text("comments"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const agentInstructions = pgTable("agent_instructions", {
  id: serial("id").primaryKey(),
  content: text("content").notNull(),
  version: integer("version").notNull().default(1),
  isActive: boolean("is_active").notNull().default(false),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertWoundAssessmentSchema = createInsertSchema(woundAssessments).omit({
  id: true,
  createdAt: true,
});

export const insertAgentInstructionsSchema = createInsertSchema(agentInstructions).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertFeedbackSchema = createInsertSchema(feedbacks).omit({
  id: true,
  createdAt: true,
});

export type InsertWoundAssessment = z.infer<typeof insertWoundAssessmentSchema>;
export type WoundAssessment = typeof woundAssessments.$inferSelect;
export type InsertFeedback = z.infer<typeof insertFeedbackSchema>;
export type Feedback = typeof feedbacks.$inferSelect;
export type InsertAgentInstructions = z.infer<typeof insertAgentInstructionsSchema>;
export type AgentInstructions = typeof agentInstructions.$inferSelect;

// Validation schemas for API endpoints
export const uploadRequestSchema = z.object({
  audience: z.enum(['family', 'patient', 'medical']),
  model: z.enum(['gpt-4o', 'gpt-3.5', 'gpt-3.5-pro', 'gemini-2.5-flash', 'gemini-2.5-pro']),
  woundOrigin: z.string().optional(),
  medicalHistory: z.string().optional(),
  woundChanges: z.string().optional(),
  currentCare: z.string().optional(),
  woundPain: z.string().optional(),
  supportAtHome: z.string().optional(),
  mobilityStatus: z.string().optional(),
  nutritionStatus: z.string().optional(),
});

export const feedbackRequestSchema = z.object({
  caseId: z.string(),
  feedbackType: z.enum(['helpful', 'not-helpful']),
  comments: z.string().optional(),
});

export type UploadRequest = z.infer<typeof uploadRequestSchema>;
export type FeedbackRequest = z.infer<typeof feedbackRequestSchema>;
