import { woundAssessments, feedbacks, agentInstructions, agentQuestions, users, companies, detectionModels, aiAnalysisModels, userProfiles, type WoundAssessment, type InsertWoundAssessment, type Feedback, type InsertFeedback, type AgentInstructions, type InsertAgentInstructions, type AgentQuestion, type InsertAgentQuestion, type User, type UpsertUser, type Company, type InsertCompany, type UserUpdate, type CompanyUpdate, type DetectionModel, type InsertDetectionModel, type AiAnalysisModel, type InsertAiAnalysisModel, type UserProfile, type InsertUserProfile } from "@shared/schema";
import { db } from "./db";
import { eq, desc, and } from "drizzle-orm";

export interface IStorage {
  // User operations - required for Replit Auth
  getUser(id: string): Promise<User | undefined>;
  upsertUser(user: UpsertUser): Promise<User>;
  getUserWoundAssessments(userId: string): Promise<WoundAssessment[]>;
  
  // User profile operations
  getUserProfile(userId: string): Promise<UserProfile | undefined>;
  createUserProfile(profile: InsertUserProfile): Promise<UserProfile>;
  updateUserProfile(userId: string, profile: Partial<InsertUserProfile>): Promise<UserProfile>;
  deleteUserProfile(userId: string): Promise<boolean>;
  
  // Wound assessment operations
  createWoundAssessment(assessment: InsertWoundAssessment): Promise<WoundAssessment>;
  getWoundAssessment(caseId: string): Promise<WoundAssessment | undefined>;
  getLatestWoundAssessment(caseId: string): Promise<WoundAssessment | undefined>;
  getWoundAssessmentHistory(caseId: string): Promise<WoundAssessment[]>;
  createFollowUpAssessment(assessment: InsertWoundAssessment): Promise<WoundAssessment>;
  findAssessmentByImageData(userId: string, imageData: string, imageSize: number): Promise<WoundAssessment | undefined>;
  deleteWoundAssessment(caseId: string): Promise<boolean>;
  createFeedback(feedback: InsertFeedback): Promise<Feedback>;
  getFeedbacksByCase(caseId: string): Promise<Feedback[]>;
  getActiveAgentInstructions(): Promise<AgentInstructions | undefined>;
  createAgentInstructions(instructions: InsertAgentInstructions): Promise<AgentInstructions>;
  updateAgentInstructions(id: number, instructions: {
    systemPrompts?: string;
    carePlanStructure?: string;
    specificWoundCare?: string;
    questionsGuidelines?: string | null;
    productRecommendations?: string | null;
  }): Promise<AgentInstructions>;
  
  // Agent question operations
  createAgentQuestion(question: InsertAgentQuestion): Promise<AgentQuestion>;
  getQuestionsBySession(sessionId: string): Promise<AgentQuestion[]>;
  answerQuestion(questionId: number, answer: string): Promise<AgentQuestion>;
  getUnansweredQuestions(sessionId: string): Promise<AgentQuestion[]>;

  // Admin operations
  getAllUsers(): Promise<User[]>;
  getUsersByCompany(companyId: number): Promise<User[]>;
  updateUser(userId: string, updates: UserUpdate): Promise<User>;
  deleteUser(userId: string): Promise<boolean>;
  getAllWoundAssessments(): Promise<WoundAssessment[]>;
  getWoundAssessmentsByUser(userId: string): Promise<WoundAssessment[]>;
  getWoundAssessmentsByDateRange(startDate: Date, endDate: Date): Promise<WoundAssessment[]>;
  
  // Company operations
  getAllCompanies(): Promise<Company[]>;
  getCompany(id: number): Promise<Company | undefined>;
  createCompany(company: InsertCompany): Promise<Company>;
  updateCompany(id: number, updates: CompanyUpdate): Promise<Company>;
  deleteCompany(id: number): Promise<boolean>;
  
  // Detection model operations
  getAllDetectionModels(): Promise<DetectionModel[]>;
  getEnabledDetectionModels(): Promise<DetectionModel[]>;
  getDetectionModel(id: number): Promise<DetectionModel | undefined>;
  createDetectionModel(model: InsertDetectionModel): Promise<DetectionModel>;
  updateDetectionModel(id: number, updates: Partial<DetectionModel>): Promise<DetectionModel>;
  toggleDetectionModel(id: number, enabled: boolean): Promise<DetectionModel>;
  deleteDetectionModel(id: number): Promise<boolean>;
  
  // AI analysis model operations
  getAllAiAnalysisModels(): Promise<AiAnalysisModel[]>;
  getEnabledAiAnalysisModels(): Promise<AiAnalysisModel[]>;
  getDefaultAiAnalysisModel(): Promise<AiAnalysisModel | undefined>;
  getAiAnalysisModel(id: number): Promise<AiAnalysisModel | undefined>;
  createAiAnalysisModel(model: InsertAiAnalysisModel): Promise<AiAnalysisModel>;
  updateAiAnalysisModel(id: number, updates: Partial<AiAnalysisModel>): Promise<AiAnalysisModel>;
  toggleAiAnalysisModel(id: number, enabled: boolean): Promise<AiAnalysisModel>;
  setDefaultAiAnalysisModel(id: number): Promise<AiAnalysisModel>;
  deleteAiAnalysisModel(id: number): Promise<boolean>;
}

export class DatabaseStorage implements IStorage {
  // Temporary in-memory storage for agent questions until database schema is updated
  private agentQuestionsStorage: Map<string, AgentQuestion[]> = new Map();
  // User operations - required for Replit Auth
  async getUser(id: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user;
  }

  async upsertUser(userData: UpsertUser): Promise<User> {
    const [user] = await db
      .insert(users)
      .values(userData)
      .onConflictDoUpdate({
        target: users.id,
        set: {
          ...userData,
          updatedAt: new Date(),
        },
      })
      .returning();
    return user;
  }

  async getUserWoundAssessments(userId: string): Promise<WoundAssessment[]> {
    return await db
      .select()
      .from(woundAssessments)
      .where(eq(woundAssessments.userId, userId))
      .orderBy(desc(woundAssessments.createdAt));
  }

  // Wound assessment operations
  async createWoundAssessment(assessment: InsertWoundAssessment): Promise<WoundAssessment> {
    const [result] = await db
      .insert(woundAssessments)
      .values(assessment)
      .returning();
    return result;
  }

  async getWoundAssessment(caseId: string): Promise<WoundAssessment | undefined> {
    const [result] = await db
      .select()
      .from(woundAssessments)
      .where(eq(woundAssessments.caseId, caseId));
    return result || undefined;
  }

  async getLatestWoundAssessment(caseId: string): Promise<WoundAssessment | undefined> {
    const [result] = await db
      .select()
      .from(woundAssessments)
      .where(eq(woundAssessments.caseId, caseId))
      .orderBy(desc(woundAssessments.versionNumber))
      .limit(1);
    return result || undefined;
  }

  async getWoundAssessmentHistory(caseId: string): Promise<WoundAssessment[]> {
    return await db
      .select()
      .from(woundAssessments)
      .where(eq(woundAssessments.caseId, caseId))
      .orderBy(desc(woundAssessments.versionNumber));
  }

  async createFollowUpAssessment(assessment: InsertWoundAssessment): Promise<WoundAssessment> {
    // Get the latest version number for this case
    const latestAssessment = await this.getLatestWoundAssessment(assessment.caseId);
    const nextVersionNumber = latestAssessment ? (latestAssessment.versionNumber + 1) : 1;
    
    const followUpAssessment = {
      ...assessment,
      versionNumber: nextVersionNumber,
      isFollowUp: true,
      previousVersion: latestAssessment?.versionNumber || null,
    };

    const [result] = await db
      .insert(woundAssessments)
      .values(followUpAssessment)
      .returning();
    return result;
  }

  async findAssessmentByImageData(userId: string, imageData: string, imageSize: number): Promise<WoundAssessment | undefined> {
    const [result] = await db
      .select()
      .from(woundAssessments)
      .where(
        and(
          eq(woundAssessments.userId, userId),
          eq(woundAssessments.imageData, imageData),
          eq(woundAssessments.imageSize, imageSize)
        )
      )
      .orderBy(desc(woundAssessments.createdAt))
      .limit(1);
    return result || undefined;
  }

  async deleteWoundAssessment(caseId: string): Promise<boolean> {
    const result = await db
      .delete(woundAssessments)
      .where(eq(woundAssessments.caseId, caseId));
    return (result.rowCount ?? 0) > 0;
  }

  async createFeedback(feedback: InsertFeedback): Promise<Feedback> {
    const [result] = await db
      .insert(feedbacks)
      .values(feedback)
      .returning();
    return result;
  }

  async getFeedbacksByCase(caseId: string): Promise<Feedback[]> {
    return await db
      .select()
      .from(feedbacks)
      .where(eq(feedbacks.caseId, caseId));
  }

  async getActiveAgentInstructions(): Promise<AgentInstructions | undefined> {
    const [result] = await db
      .select()
      .from(agentInstructions)
      .where(eq(agentInstructions.isActive, true))
      .orderBy(desc(agentInstructions.version))
      .limit(1);
    return result || undefined;
  }

  async createAgentInstructions(instructions: InsertAgentInstructions): Promise<AgentInstructions> {
    // Deactivate all previous instructions
    await db
      .update(agentInstructions)
      .set({ isActive: false })
      .where(eq(agentInstructions.isActive, true));

    // Create new active instructions
    const [result] = await db
      .insert(agentInstructions)
      .values({ ...instructions, isActive: true })
      .returning();
    return result;
  }

  async updateAgentInstructions(id: number, instructions: {
    systemPrompts?: string;
    carePlanStructure?: string;
    specificWoundCare?: string;
    questionsGuidelines?: string | null;
    productRecommendations?: string | null;
  }): Promise<AgentInstructions> {
    const [result] = await db
      .update(agentInstructions)
      .set({ 
        ...instructions,
        updatedAt: new Date() 
      })
      .where(eq(agentInstructions.id, id))
      .returning();
    return result;
  }

  // Agent question operations (using temporary in-memory storage)
  async createAgentQuestion(question: InsertAgentQuestion): Promise<AgentQuestion> {
    const sessionQuestions = this.agentQuestionsStorage.get(question.sessionId) || [];
    const newQuestion: AgentQuestion = {
      id: sessionQuestions.length + 1,
      userId: question.userId,
      sessionId: question.sessionId,
      caseId: question.caseId || null,
      question: question.question,
      answer: null,
      questionType: question.questionType,
      isAnswered: false,
      context: question.context || null,
      createdAt: new Date(),
      answeredAt: null,
    };
    sessionQuestions.push(newQuestion);
    this.agentQuestionsStorage.set(question.sessionId, sessionQuestions);
    return newQuestion;
  }

  async getQuestionsBySession(sessionId: string): Promise<AgentQuestion[]> {
    return this.agentQuestionsStorage.get(sessionId) || [];
  }

  async answerQuestion(questionId: number, answer: string): Promise<AgentQuestion> {
    const sessionIds = Array.from(this.agentQuestionsStorage.keys());
    for (const sessionId of sessionIds) {
      const questions = this.agentQuestionsStorage.get(sessionId) || [];
      const question = questions.find(q => q.id === questionId);
      if (question) {
        question.answer = answer;
        question.isAnswered = true;
        question.answeredAt = new Date();
        return question;
      }
    }
    throw new Error(`Question with ID ${questionId} not found`);
  }

  async getUnansweredQuestions(sessionId: string): Promise<AgentQuestion[]> {
    const questions = this.agentQuestionsStorage.get(sessionId) || [];
    return questions.filter(q => !q.isAnswered);
  }

  // Admin operations implementation
  async getAllUsers(): Promise<User[]> {
    return await db.select().from(users).orderBy(desc(users.createdAt));
  }

  async getUsersByCompany(companyId: number): Promise<User[]> {
    return await db.select().from(users).where(eq(users.companyId, companyId)).orderBy(desc(users.createdAt));
  }

  async updateUser(userId: string, updates: UserUpdate): Promise<User> {
    const [user] = await db
      .update(users)
      .set({
        ...updates,
        updatedAt: new Date(),
      })
      .where(eq(users.id, userId))
      .returning();
    return user;
  }

  async deleteUser(userId: string): Promise<boolean> {
    const result = await db.delete(users).where(eq(users.id, userId));
    return result.rowCount !== null && result.rowCount > 0;
  }

  async getAllWoundAssessments(): Promise<WoundAssessment[]> {
    return await db.select().from(woundAssessments).orderBy(desc(woundAssessments.createdAt));
  }

  async getWoundAssessmentsByUser(userId: string): Promise<WoundAssessment[]> {
    return await db.select().from(woundAssessments).where(eq(woundAssessments.userId, userId)).orderBy(desc(woundAssessments.createdAt));
  }

  async getWoundAssessmentsByDateRange(startDate: Date, endDate: Date): Promise<WoundAssessment[]> {
    return await db.select().from(woundAssessments)
      .where(
        // Since we're using timestamp columns, we need to handle date comparison properly
        // This is a simple implementation - in production you'd want proper date handling
        desc(woundAssessments.createdAt)
      )
      .orderBy(desc(woundAssessments.createdAt));
  }

  // Company operations implementation
  async getAllCompanies(): Promise<Company[]> {
    return await db.select().from(companies).orderBy(desc(companies.createdAt));
  }

  async getCompany(id: number): Promise<Company | undefined> {
    const [company] = await db.select().from(companies).where(eq(companies.id, id));
    return company;
  }

  async createCompany(company: InsertCompany): Promise<Company> {
    const [newCompany] = await db.insert(companies).values(company).returning();
    return newCompany;
  }

  async updateCompany(id: number, updates: CompanyUpdate): Promise<Company> {
    const [company] = await db
      .update(companies)
      .set({
        ...updates,
        updatedAt: new Date(),
      })
      .where(eq(companies.id, id))
      .returning();
    return company;
  }

  async deleteCompany(id: number): Promise<boolean> {
    const result = await db.delete(companies).where(eq(companies.id, id));
    return result.rowCount !== null && result.rowCount > 0;
  }

  // Detection model operations implementation
  async getAllDetectionModels(): Promise<DetectionModel[]> {
    return await db.select().from(detectionModels).orderBy(desc(detectionModels.priority), detectionModels.name);
  }

  async getEnabledDetectionModels(): Promise<DetectionModel[]> {
    return await db.select().from(detectionModels)
      .where(eq(detectionModels.isEnabled, true))
      .orderBy(desc(detectionModels.priority), detectionModels.name);
  }

  async getDetectionModel(id: number): Promise<DetectionModel | undefined> {
    const [model] = await db.select().from(detectionModels).where(eq(detectionModels.id, id));
    return model;
  }

  async createDetectionModel(model: InsertDetectionModel): Promise<DetectionModel> {
    const [newModel] = await db.insert(detectionModels).values(model).returning();
    return newModel;
  }

  async updateDetectionModel(id: number, updates: Partial<DetectionModel>): Promise<DetectionModel> {
    const [model] = await db
      .update(detectionModels)
      .set({
        ...updates,
        updatedAt: new Date(),
      })
      .where(eq(detectionModels.id, id))
      .returning();
    return model;
  }

  async toggleDetectionModel(id: number, enabled: boolean): Promise<DetectionModel> {
    const [model] = await db
      .update(detectionModels)
      .set({
        isEnabled: enabled,
        updatedAt: new Date(),
      })
      .where(eq(detectionModels.id, id))
      .returning();
    return model;
  }

  async deleteDetectionModel(id: number): Promise<boolean> {
    const result = await db.delete(detectionModels).where(eq(detectionModels.id, id));
    return result.rowCount !== null && result.rowCount > 0;
  }

  // AI analysis model operations implementation
  async getAllAiAnalysisModels(): Promise<AiAnalysisModel[]> {
    return await db.select().from(aiAnalysisModels).orderBy(desc(aiAnalysisModels.priority), aiAnalysisModels.name);
  }

  async getEnabledAiAnalysisModels(): Promise<AiAnalysisModel[]> {
    return await db.select().from(aiAnalysisModels)
      .where(eq(aiAnalysisModels.isEnabled, true))
      .orderBy(desc(aiAnalysisModels.priority), aiAnalysisModels.name);
  }

  async getDefaultAiAnalysisModel(): Promise<AiAnalysisModel | undefined> {
    const [model] = await db.select().from(aiAnalysisModels)
      .where(and(eq(aiAnalysisModels.isDefault, true), eq(aiAnalysisModels.isEnabled, true)));
    return model;
  }

  async getAiAnalysisModel(id: number): Promise<AiAnalysisModel | undefined> {
    const [model] = await db.select().from(aiAnalysisModels).where(eq(aiAnalysisModels.id, id));
    return model;
  }

  async createAiAnalysisModel(model: InsertAiAnalysisModel): Promise<AiAnalysisModel> {
    const [newModel] = await db.insert(aiAnalysisModels).values(model).returning();
    return newModel;
  }

  async updateAiAnalysisModel(id: number, updates: Partial<AiAnalysisModel>): Promise<AiAnalysisModel> {
    const [model] = await db
      .update(aiAnalysisModels)
      .set({
        ...updates,
        updatedAt: new Date(),
      })
      .where(eq(aiAnalysisModels.id, id))
      .returning();
    return model;
  }

  async toggleAiAnalysisModel(id: number, enabled: boolean): Promise<AiAnalysisModel> {
    const [model] = await db
      .update(aiAnalysisModels)
      .set({
        isEnabled: enabled,
        updatedAt: new Date(),
      })
      .where(eq(aiAnalysisModels.id, id))
      .returning();
    return model;
  }

  async setDefaultAiAnalysisModel(id: number): Promise<AiAnalysisModel> {
    // First, unset all defaults
    await db
      .update(aiAnalysisModels)
      .set({
        isDefault: false,
        updatedAt: new Date(),
      });

    // Then set the new default
    const [model] = await db
      .update(aiAnalysisModels)
      .set({
        isDefault: true,
        isEnabled: true, // Ensure the default model is enabled
        updatedAt: new Date(),
      })
      .where(eq(aiAnalysisModels.id, id))
      .returning();
    return model;
  }

  async deleteAiAnalysisModel(id: number): Promise<boolean> {
    const result = await db.delete(aiAnalysisModels).where(eq(aiAnalysisModels.id, id));
    return result.rowCount !== null && result.rowCount > 0;
  }

  // User profile operations
  async getUserProfile(userId: string): Promise<UserProfile | undefined> {
    const [profile] = await db.select().from(userProfiles).where(eq(userProfiles.userId, userId));
    return profile;
  }

  async createUserProfile(profile: InsertUserProfile): Promise<UserProfile> {
    const [newProfile] = await db
      .insert(userProfiles)
      .values(profile)
      .returning();
    return newProfile;
  }

  async updateUserProfile(userId: string, profile: Partial<InsertUserProfile>): Promise<UserProfile> {
    const [updatedProfile] = await db
      .update(userProfiles)
      .set({
        ...profile,
        updatedAt: new Date(),
      })
      .where(eq(userProfiles.userId, userId))
      .returning();
    return updatedProfile;
  }

  async deleteUserProfile(userId: string): Promise<boolean> {
    const result = await db.delete(userProfiles).where(eq(userProfiles.userId, userId));
    return result.rowCount !== null && result.rowCount > 0;
  }
}

export const storage = new DatabaseStorage();