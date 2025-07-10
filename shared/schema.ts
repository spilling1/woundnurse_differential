import { pgTable, text, serial, integer, timestamp, json, jsonb, boolean, varchar, index } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Session storage table for Replit Auth
export const sessions = pgTable(
  "sessions",
  {
    sid: varchar("sid").primaryKey(),
    sess: jsonb("sess").notNull(),
    expire: timestamp("expire").notNull(),
  },
  (table) => [index("IDX_session_expire").on(table.expire)],
);

// Companies table for future multi-tenant support
export const companies = pgTable("companies", {
  id: serial("id").primaryKey(),
  name: varchar("name").notNull(),
  domain: varchar("domain").unique(),
  status: varchar("status").notNull().default("active"), // 'active' | 'inactive' | 'suspended'
  maxUsers: integer("max_users").default(100),
  subscriptionPlan: varchar("subscription_plan").default("basic"), // 'basic' | 'pro' | 'enterprise'
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// User storage table for custom authentication
export const users = pgTable("users", {
  id: varchar("id").primaryKey().notNull(),
  email: varchar("email").unique().notNull(),
  password: varchar("password"), // bcrypt hashed password
  firstName: varchar("first_name"),
  lastName: varchar("last_name"),
  profileImageUrl: varchar("profile_image_url"),
  role: varchar("role").notNull().default("user"), // 'admin' | 'user' | 'nurse' | 'manager'
  status: varchar("status").notNull().default("active"), // 'active' | 'inactive' | 'suspended'
  companyId: integer("company_id").references(() => companies.id),
  lastLoginAt: timestamp("last_login_at"),
  mustChangePassword: boolean("must_change_password").default(false),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// User profiles table for medical and personal information
export const userProfiles = pgTable("user_profiles", {
  id: serial("id").primaryKey(),
  userId: varchar("user_id").notNull().references(() => users.id, { onDelete: "cascade" }),
  
  // Profile type and basic info
  profileType: varchar("profile_type").notNull(), // "patient" or "caregiver"
  age: integer("age"),
  gender: varchar("gender"),
  
  // Medical conditions (for patients)
  isDiabetic: boolean("is_diabetic").default(false),
  hasHighBloodPressure: boolean("has_high_blood_pressure").default(false),
  hasHeartDisease: boolean("has_heart_disease").default(false),
  hasKidneyDisease: boolean("has_kidney_disease").default(false),
  hasCirculationIssues: boolean("has_circulation_issues").default(false),
  hasOstomy: boolean("has_ostomy").default(false),
  ostomyType: varchar("ostomy_type"), // if has_ostomy is true
  
  // Mobility and lifestyle
  mobilityStatus: varchar("mobility_status"), // "fully_mobile", "limited_mobility", "wheelchair", "bed_ridden"
  assistiveDevices: text("assistive_devices").array(), // walker, crutches, etc.
  activityLevel: varchar("activity_level"), // "very_active", "moderately_active", "sedentary"
  
  // Medications and allergies
  currentMedications: text("current_medications"), // free text for now
  medicationAllergies: text("medication_allergies"),
  otherAllergies: text("other_allergies"),
  
  // Nutrition and lifestyle
  nutritionStatus: varchar("nutrition_status"), // "excellent", "good", "fair", "poor"
  dietRestrictions: text("diet_restrictions"),
  smokingStatus: varchar("smoking_status"), // "never", "former", "current"
  alcoholUse: varchar("alcohol_use"), // "none", "occasional", "moderate", "heavy"
  
  // Care goals and preferences
  primaryGoals: text("primary_goals").array(), // quality_of_life, pain_management, independence, etc.
  carePreferences: text("care_preferences"),
  emergencyContact: varchar("emergency_contact"),
  
  // Caregiver-specific information
  relationshipToPatient: varchar("relationship_to_patient"), // for caregivers
  caregivingExperience: varchar("caregiving_experience"), // "none", "some", "experienced"
  professionalBackground: varchar("professional_background"), // "medical", "non_medical", "retired_medical"
  
  // System fields
  profileCompleted: boolean("profile_completed").default(false),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const woundAssessments = pgTable("wound_assessments", {
  id: serial("id").primaryKey(),
  caseId: text("case_id").notNull(),
  caseName: text("case_name"), // Optional user-defined name for the case
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
  classification: jsonb("classification"), // wound type, stage, size, etc.
  detectionData: jsonb("detection_data"), // YOLO9 detection results, measurements, bounding boxes
  carePlan: text("care_plan").notNull(),
  
  // Follow-up and versioning
  version: text("version").notNull().default("1"), // Keep original version field
  versionNumber: integer("version_number").notNull().default(1), // Version number for this case
  isFollowUp: boolean("is_follow_up").notNull().default(false),
  previousVersion: integer("previous_version"), // Reference to previous version
  progressNotes: text("progress_notes"), // Patient-reported progress since last assessment
  treatmentResponse: text("treatment_response"), // How wound responded to previous treatment
  contextData: jsonb("context_data"), // Store all questionnaire data as JSON
  
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
  systemPrompts: text("system_prompts").notNull(),
  carePlanStructure: text("care_plan_structure").notNull(),
  specificWoundCare: text("specific_wound_care").notNull(),
  questionsGuidelines: text("questions_guidelines"),
  productRecommendations: text("product_recommendations"),
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

// Detection Models Configuration
export const detectionModels = pgTable("detection_models", {
  id: serial("id").primaryKey(),
  name: varchar("name").notNull(), // 'yolo9', 'google-cloud-vision', 'azure-computer-vision', 'cnn-classifier', 'enhanced-fallback'
  displayName: varchar("display_name").notNull(), // 'YOLO9 Detection', 'Google Cloud Vision', etc.
  description: text("description").notNull(),
  isEnabled: boolean("is_enabled").notNull().default(true),
  priority: integer("priority").notNull().default(0), // Higher priority = tried first
  endpoint: varchar("endpoint"), // API endpoint or service URL
  requiresApiKey: boolean("requires_api_key").notNull().default(false),
  apiKeyName: varchar("api_key_name"), // Environment variable name for API key
  modelType: varchar("model_type").notNull(), // 'yolo', 'cloud-api', 'cnn', 'fallback'
  capabilities: jsonb("capabilities").notNull(), // JSON array of capabilities: ['detection', 'measurements', 'classification']
  config: jsonb("config"), // Model-specific configuration
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// AI Analysis Models Configuration
export const aiAnalysisModels = pgTable("ai_analysis_models", {
  id: serial("id").primaryKey(),
  name: varchar("name").notNull(), // 'gpt-4o', 'gpt-3.5', 'gemini-2.5-pro', etc.
  displayName: varchar("display_name").notNull(), // 'GPT-4o', 'Gemini 2.5 Pro', etc.
  description: text("description").notNull(),
  isEnabled: boolean("is_enabled").notNull().default(true),
  isDefault: boolean("is_default").notNull().default(false), // Only one can be default
  priority: integer("priority").notNull().default(0), // Higher priority = preferred
  provider: varchar("provider").notNull(), // 'openai', 'google', 'anthropic'
  modelId: varchar("model_id").notNull(), // 'gpt-4o', 'gemini-2.5-pro', etc.
  requiresApiKey: boolean("requires_api_key").notNull().default(true),
  apiKeyName: varchar("api_key_name"), // Environment variable name for API key
  capabilities: jsonb("capabilities").notNull(), // JSON array: ['vision', 'text', 'reasoning']
  config: jsonb("config"), // Model-specific configuration like temperature, max_tokens
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Product recommendations table
export const productRecommendations = pgTable("product_recommendations", {
  id: serial("id").primaryKey(),
  name: varchar("name").notNull(), // e.g., "Hydrocolloid Dressings"
  category: varchar("category").notNull(), // e.g., "wound_dressing", "cleansing", "moisturizing", "compression", "protection"
  description: text("description"),
  amazonSearchUrl: text("amazon_search_url"), // Full Amazon search URL
  searchKeywords: text("search_keywords"), // Keywords for Amazon search
  woundTypes: text("wound_types").array(), // e.g., ["venous_ulcer", "diabetic_ulcer", "pressure_ulcer"]
  audiences: text("audiences").array(), // e.g., ["family", "patient", "professional"]
  priority: integer("priority").default(50), // 1-100, higher numbers = higher priority
  isActive: boolean("is_active").notNull().default(true),
  timesRecommended: integer("times_recommended").default(0),
  extractedFromCaseId: varchar("extracted_from_case_id"), // Track which case this was first extracted from
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// AI interactions table for recording all AI prompts and responses
export const aiInteractions = pgTable("ai_interactions", {
  id: serial("id").primaryKey(),
  caseId: varchar("case_id").notNull(), // Link to wound assessment case
  stepType: varchar("step_type").notNull(), // 'independent_classification', 'yolo_reconsideration', 'question_generation', 'care_plan_generation'
  modelUsed: varchar("model_used").notNull(), // 'gemini-2.5-pro', 'gpt-4o', etc.
  promptSent: text("prompt_sent").notNull(), // Full prompt sent to AI
  responseReceived: text("response_received").notNull(), // Full response from AI
  parsedResult: jsonb("parsed_result"), // Parsed JSON result if applicable
  processingTimeMs: integer("processing_time_ms"), // Time taken for this interaction
  confidenceScore: integer("confidence_score"), // AI confidence (0-100)
  errorOccurred: boolean("error_occurred").default(false),
  errorMessage: text("error_message"), // Error details if any
  createdAt: timestamp("created_at").defaultNow(),
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

export const insertDetectionModelSchema = createInsertSchema(detectionModels).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertAiAnalysisModelSchema = createInsertSchema(aiAnalysisModels).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertCompanySchema = createInsertSchema(companies).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertUserProfileSchema = createInsertSchema(userProfiles).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertProductRecommendationSchema = createInsertSchema(productRecommendations).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

// Wound Types Configuration
export const woundTypes = pgTable("wound_types", {
  id: serial("id").primaryKey(),
  name: varchar("name").notNull(), // 'pressure_injury', 'venous_ulcer', 'arterial_insufficiency_ulcer', etc.
  displayName: varchar("display_name").notNull(), // 'Pressure Injury (Ulcer)', 'Venous Ulcer', etc.
  description: text("description"),
  instructions: text("instructions").notNull(), // AI instructions for this wound type
  synonyms: text("synonyms").array(), // Alternative names/synonyms for this wound type
  isEnabled: boolean("is_enabled").notNull().default(true),
  isDefault: boolean("is_default").notNull().default(false), // For "General Instructions"
  priority: integer("priority").notNull().default(0), // Higher priority = tried first
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertAiInteractionSchema = createInsertSchema(aiInteractions).omit({
  id: true,
  createdAt: true,
});

export const insertWoundTypeSchema = createInsertSchema(woundTypes).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertWoundAssessment = z.infer<typeof insertWoundAssessmentSchema>;
export type WoundAssessment = typeof woundAssessments.$inferSelect;
export type InsertFeedback = z.infer<typeof insertFeedbackSchema>;
export type Feedback = typeof feedbacks.$inferSelect;
export type InsertAgentInstructions = z.infer<typeof insertAgentInstructionsSchema>;
export type AgentInstructions = typeof agentInstructions.$inferSelect;
export type InsertAgentQuestion = z.infer<typeof insertAgentQuestionSchema>;
export type AgentQuestion = typeof agentQuestions.$inferSelect;
export type InsertDetectionModel = z.infer<typeof insertDetectionModelSchema>;
export type DetectionModel = typeof detectionModels.$inferSelect;
export type InsertAiAnalysisModel = z.infer<typeof insertAiAnalysisModelSchema>;
export type AiAnalysisModel = typeof aiAnalysisModels.$inferSelect;
export type InsertCompany = z.infer<typeof insertCompanySchema>;
export type Company = typeof companies.$inferSelect;
export type UpsertUser = typeof users.$inferInsert;
export type User = typeof users.$inferSelect;
export type InsertUserProfile = z.infer<typeof insertUserProfileSchema>;
export type UserProfile = typeof userProfiles.$inferSelect;
export type InsertProductRecommendation = z.infer<typeof insertProductRecommendationSchema>;
export type ProductRecommendation = typeof productRecommendations.$inferSelect;
export type InsertWoundType = z.infer<typeof insertWoundTypeSchema>;
export type WoundType = typeof woundTypes.$inferSelect;

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

// Wound Classification interface for frontend types
export interface WoundClassification {
  woundType: string;
  stage: string;
  size: string;
  woundBed: string;
  exudate: string;
  infectionSigns: string[];
  location: string;
  additionalObservations: string;
  confidence: number;
  classificationMethod?: string;
  modelInfo?: {
    type: string;
    accuracy: string;
    apiCall?: boolean;
    processingTime?: number;
  };
  detectionMetadata?: {
    model: string;
    version: string;
    processingTime?: number;
    multipleWounds: boolean;
  };
  detection?: {
    confidence: number;
    boundingBox: any;
    measurements: any;
    scaleCalibrated: boolean;
  };
  preciseMeasurements?: any;
  imageAnalysis?: string;
  multipleWounds?: boolean;
}

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

// Admin validation schemas
export const userUpdateSchema = z.object({
  role: z.enum(['admin', 'user', 'nurse', 'manager']).optional(),
  status: z.enum(['active', 'inactive', 'suspended']).optional(),
  companyId: z.number().nullable().optional(),
  password: z.string().optional(),
  mustChangePassword: z.boolean().optional(),
  lastLoginAt: z.date().optional(),
});

export const companyCreateSchema = z.object({
  name: z.string().min(1, "Company name is required"),
  domain: z.string().optional(),
  maxUsers: z.number().positive().default(100),
  subscriptionPlan: z.enum(['basic', 'pro', 'enterprise']).default('basic'),
});

export const companyUpdateSchema = z.object({
  name: z.string().min(1).optional(),
  domain: z.string().optional(),
  status: z.enum(['active', 'inactive', 'suspended']).optional(),
  maxUsers: z.number().positive().optional(),
  subscriptionPlan: z.enum(['basic', 'pro', 'enterprise']).optional(),
});

export type UserUpdate = z.infer<typeof userUpdateSchema>;
export type CompanyCreate = z.infer<typeof companyCreateSchema>;
export type CompanyUpdate = z.infer<typeof companyUpdateSchema>;
