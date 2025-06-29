import { pgTable, text, serial, integer, timestamp, json, boolean, varchar, index } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Session storage table for Replit Auth
export const sessions = pgTable(
  "sessions",
  {
    sid: varchar("sid").primaryKey(),
    sess: json("sess").notNull(),
    expire: timestamp("expire").notNull(),
  },
  (table) => [index("IDX_session_expire").on(table.expire)],
);

// User storage table for Replit Auth
export const users = pgTable("users", {
  id: varchar("id").primaryKey().notNull(),
  email: varchar("email").unique(),
  firstName: varchar("first_name"),
  lastName: varchar("last_name"),
  profileImageUrl: varchar("profile_image_url"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const woundAssessments = pgTable("wound_assessments", {
  id: serial("id").primaryKey(),
  caseId: text("case_id").notNull(),
  userId: varchar("user_id"), // Optional - for logged-in users
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
  
  // Follow-up and versioning
  version: text("version").notNull().default("1"), // Keep original version field
  versionNumber: integer("version_number").notNull().default(1), // Version number for this case
  isFollowUp: boolean("is_follow_up").notNull().default(false),
  previousVersion: integer("previous_version"), // Reference to previous version
  progressNotes: text("progress_notes"), // Patient-reported progress since last assessment
  treatmentResponse: text("treatment_response"), // How wound responded to previous treatment
  contextData: json("context_data"), // Store all questionnaire data as JSON
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  // Composite index for querying latest version of each case
  caseVersionIndex: index("case_version_idx").on(table.caseId, table.versionNumber),
}));

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

export const agentQuestions = pgTable("agent_questions", {
  id: serial("id").primaryKey(),
  sessionId: varchar("session_id").notNull(), // Unique session ID for Q&A
  caseId: varchar("case_id"),
  userId: varchar("user_id").notNull(),
  question: text("question").notNull(),
  answer: text("answer"),
  questionType: varchar("question_type").notNull(), // 'clarification', 'medical_history', 'symptom_detail', etc.
  isAnswered: boolean("is_answered").default(false),
  context: text("context"), // Additional context for the question as JSON string
  createdAt: timestamp("created_at").defaultNow(),
  answeredAt: timestamp("answered_at"),
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

export const insertAgentQuestionSchema = createInsertSchema(agentQuestions).omit({
  id: true,
  createdAt: true,
  answeredAt: true,
});

export type InsertWoundAssessment = z.infer<typeof insertWoundAssessmentSchema>;
export type WoundAssessment = typeof woundAssessments.$inferSelect;
export type InsertFeedback = z.infer<typeof insertFeedbackSchema>;
export type Feedback = typeof feedbacks.$inferSelect;
export type InsertAgentInstructions = z.infer<typeof insertAgentInstructionsSchema>;
export type AgentInstructions = typeof agentInstructions.$inferSelect;
export type InsertAgentQuestion = z.infer<typeof insertAgentQuestionSchema>;
export type AgentQuestion = typeof agentQuestions.$inferSelect;
export type UpsertUser = typeof users.$inferInsert;
export type User = typeof users.$inferSelect;

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

export const followUpRequestSchema = z.object({
  caseId: z.string(),
  model: z.enum(['gpt-4o', 'gpt-3.5', 'gpt-3.5-pro', 'gemini-2.5-flash', 'gemini-2.5-pro']),
  progressNotes: z.string(),
  treatmentResponse: z.string(),
  additionalInfo: z.string().optional(),
  woundOrigin: z.string().optional(),
  medicalHistory: z.string().optional(),
  woundChanges: z.string().optional(),
  currentCare: z.string().optional(),
  woundPain: z.string().optional(),
  supportAtHome: z.string().optional(),
  mobilityStatus: z.string().optional(),
  nutritionStatus: z.string().optional(),
});

export type UploadRequest = z.infer<typeof uploadRequestSchema>;
export type FeedbackRequest = z.infer<typeof feedbackRequestSchema>;
export type FollowUpRequest = z.infer<typeof followUpRequestSchema>;
