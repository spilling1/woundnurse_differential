import { pgTable, text, serial, integer, timestamp, json } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const woundAssessments = pgTable("wound_assessments", {
  id: serial("id").primaryKey(),
  caseId: text("case_id").notNull().unique(),
  audience: text("audience").notNull(), // 'family' | 'patient' | 'medical'
  model: text("model").notNull(), // 'gpt-4o' | 'gpt-3.5' | 'gpt-3.5-pro'
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

export const insertWoundAssessmentSchema = createInsertSchema(woundAssessments).omit({
  id: true,
  createdAt: true,
});

export const insertFeedbackSchema = createInsertSchema(feedbacks).omit({
  id: true,
  createdAt: true,
});

export type InsertWoundAssessment = z.infer<typeof insertWoundAssessmentSchema>;
export type WoundAssessment = typeof woundAssessments.$inferSelect;
export type InsertFeedback = z.infer<typeof insertFeedbackSchema>;
export type Feedback = typeof feedbacks.$inferSelect;

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
