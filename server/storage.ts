import { woundAssessments, feedbacks, agentInstructions, agentQuestions, users, companies, detectionModels, aiAnalysisModels, userProfiles, productRecommendations, aiInteractions, woundTypes, type WoundAssessment, type InsertWoundAssessment, type Feedback, type InsertFeedback, type AgentInstructions, type InsertAgentInstructions, type AgentQuestion, type InsertAgentQuestion, type User, type UpsertUser, type Company, type InsertCompany, type UserUpdate, type CompanyUpdate, type DetectionModel, type InsertDetectionModel, type AiAnalysisModel, type InsertAiAnalysisModel, type UserProfile, type InsertUserProfile, type ProductRecommendation, type InsertProductRecommendation, type WoundType, type InsertWoundType } from "@shared/schema";
import { db } from "./db";
import { eq, desc, and } from "drizzle-orm";

export interface IStorage {
  // User operations - required for custom auth
  getUser(id: string): Promise<User | undefined>;
  getUserByEmail(email: string): Promise<User | undefined>;
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
  updateWoundAssessment(caseId: string, versionNumber: number, updates: Partial<WoundAssessment>): Promise<WoundAssessment>;
  updateCaseName(caseId: string, caseName: string): Promise<void>;
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
    duplicateDetectionEnabled?: boolean;
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

  // Product recommendation operations
  getAllProductRecommendations(): Promise<ProductRecommendation[]>;
  getActiveProductRecommendations(): Promise<ProductRecommendation[]>;
  getProductRecommendationsByCategory(category: string): Promise<ProductRecommendation[]>;
  getProductRecommendationsByWoundType(woundType: string): Promise<ProductRecommendation[]>;
  getProductRecommendation(id: number): Promise<ProductRecommendation | undefined>;
  createProductRecommendation(product: InsertProductRecommendation): Promise<ProductRecommendation>;
  updateProductRecommendation(id: number, updates: Partial<ProductRecommendation>): Promise<ProductRecommendation>;
  toggleProductRecommendation(id: number, isActive: boolean): Promise<ProductRecommendation>;
  incrementProductRecommendationUsage(id: number): Promise<ProductRecommendation>;
  deleteProductRecommendation(id: number): Promise<boolean>;
  
  // AI interaction logging operations
  createAiInteraction(interaction: {
    caseId: string;
    stepType: string;
    modelUsed: string;
    promptSent: string;
    responseReceived: string;
    parsedResult?: any;
    processingTimeMs?: number;
    confidenceScore?: number;
    errorOccurred?: boolean;
    errorMessage?: string;
  }): Promise<void>;
  getAiInteractionsByCase(caseId: string): Promise<any[]>;
  getAllAiInteractions(): Promise<any[]>;
  
  // Wound type operations
  getAllWoundTypes(): Promise<WoundType[]>;
  getEnabledWoundTypes(): Promise<WoundType[]>;
  getWoundType(id: number): Promise<WoundType | undefined>;
  getWoundTypeByName(name: string): Promise<WoundType | undefined>;
  createWoundType(woundType: InsertWoundType): Promise<WoundType>;
  updateWoundType(id: number, updates: Partial<WoundType>): Promise<WoundType>;
  toggleWoundType(id: number, enabled: boolean): Promise<WoundType>;
  deleteWoundType(id: number): Promise<boolean>;
  getDefaultWoundType(): Promise<WoundType | undefined>;
  setDefaultWoundType(id: number): Promise<WoundType>;
}

export class DatabaseStorage implements IStorage {
  // Temporary in-memory storage for agent questions until database schema is updated
  private agentQuestionsStorage: Map<string, AgentQuestion[]> = new Map();
  // User operations - required for Replit Auth
  async getUser(id: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user;
  }

  async getUserByEmail(email: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.email, email));
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

  async updateWoundAssessment(caseId: string, versionNumber: number, updates: Partial<WoundAssessment>): Promise<WoundAssessment> {
    const [result] = await db
      .update(woundAssessments)
      .set(updates)
      .where(and(
        eq(woundAssessments.caseId, caseId),
        eq(woundAssessments.versionNumber, versionNumber)
      ))
      .returning();
    return result;
  }

  async updateCaseName(caseId: string, caseName: string): Promise<void> {
    await db
      .update(woundAssessments)
      .set({ caseName })
      .where(eq(woundAssessments.caseId, caseId));
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
    duplicateDetectionEnabled?: boolean;
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

  // Agent question operations (using database storage)
  async createAgentQuestion(question: InsertAgentQuestion): Promise<AgentQuestion> {
    const newQuestion = await db.insert(agentQuestions).values({
      userId: question.userId,
      sessionId: question.sessionId,
      caseId: question.caseId || null,
      question: question.question,
      answer: null,
      questionType: question.questionType,
      isAnswered: false,
      context: question.context || null,
    }).returning();
    
    return newQuestion[0];
  }

  async getQuestionsBySession(sessionId: string): Promise<AgentQuestion[]> {
    return await db.select().from(agentQuestions).where(eq(agentQuestions.sessionId, sessionId));
  }

  async answerQuestion(questionId: number, answer: string): Promise<AgentQuestion> {
    const updated = await db.update(agentQuestions)
      .set({ 
        answer, 
        isAnswered: true, 
        answeredAt: new Date() 
      })
      .where(eq(agentQuestions.id, questionId))
      .returning();
    
    if (updated.length === 0) {
      throw new Error(`Question with ID ${questionId} not found`);
    }
    
    return updated[0];
  }

  async getUnansweredQuestions(sessionId: string): Promise<AgentQuestion[]> {
    return await db.select().from(agentQuestions)
      .where(and(
        eq(agentQuestions.sessionId, sessionId),
        eq(agentQuestions.isAnswered, false)
      ));
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

  // Product recommendation operations
  async getAllProductRecommendations(): Promise<ProductRecommendation[]> {
    return await db.select().from(productRecommendations).orderBy(desc(productRecommendations.priority), productRecommendations.name);
  }

  async getActiveProductRecommendations(): Promise<ProductRecommendation[]> {
    return await db.select().from(productRecommendations)
      .where(eq(productRecommendations.isActive, true))
      .orderBy(desc(productRecommendations.priority), productRecommendations.name);
  }

  async getProductRecommendationsByCategory(category: string): Promise<ProductRecommendation[]> {
    return await db.select().from(productRecommendations)
      .where(and(eq(productRecommendations.category, category), eq(productRecommendations.isActive, true)))
      .orderBy(desc(productRecommendations.priority), productRecommendations.name);
  }

  async getProductRecommendationsByWoundType(woundType: string): Promise<ProductRecommendation[]> {
    return await db.select().from(productRecommendations)
      .where(and(eq(productRecommendations.isActive, true)))
      .orderBy(desc(productRecommendations.priority), productRecommendations.name);
    // Note: Array contains check would need SQL function - simplified for now
  }

  async getProductRecommendation(id: number): Promise<ProductRecommendation | undefined> {
    const [product] = await db.select().from(productRecommendations).where(eq(productRecommendations.id, id));
    return product || undefined;
  }

  async createProductRecommendation(product: InsertProductRecommendation): Promise<ProductRecommendation> {
    const [newProduct] = await db.insert(productRecommendations).values(product).returning();
    return newProduct;
  }

  async updateProductRecommendation(id: number, updates: Partial<ProductRecommendation>): Promise<ProductRecommendation> {
    const [updated] = await db.update(productRecommendations)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(productRecommendations.id, id))
      .returning();
    return updated;
  }

  async toggleProductRecommendation(id: number, isActive: boolean): Promise<ProductRecommendation> {
    const [updated] = await db.update(productRecommendations)
      .set({ isActive, updatedAt: new Date() })
      .where(eq(productRecommendations.id, id))
      .returning();
    return updated;
  }

  async incrementProductRecommendationUsage(id: number): Promise<ProductRecommendation> {
    const product = await this.getProductRecommendation(id);
    if (!product) throw new Error('Product not found');
    
    const [updated] = await db.update(productRecommendations)
      .set({ 
        timesRecommended: (product.timesRecommended || 0) + 1, 
        updatedAt: new Date() 
      })
      .where(eq(productRecommendations.id, id))
      .returning();
    return updated;
  }

  async deleteProductRecommendation(id: number): Promise<boolean> {
    const [deleted] = await db.delete(productRecommendations).where(eq(productRecommendations.id, id)).returning();
    return !!deleted;
  }

  // AI interaction logging operations
  async createAiInteraction(interaction: {
    caseId: string;
    stepType: string;
    modelUsed: string;
    promptSent: string;
    responseReceived: string;
    parsedResult?: any;
    processingTimeMs?: number;
    confidenceScore?: number;
    errorOccurred?: boolean;
    errorMessage?: string;
  }): Promise<void> {
    await db.insert(aiInteractions).values({
      caseId: interaction.caseId,
      stepType: interaction.stepType,
      modelUsed: interaction.modelUsed,
      promptSent: interaction.promptSent,
      responseReceived: interaction.responseReceived,
      parsedResult: interaction.parsedResult || null,
      processingTimeMs: interaction.processingTimeMs || null,
      confidenceScore: interaction.confidenceScore || null,
      errorOccurred: interaction.errorOccurred || false,
      errorMessage: interaction.errorMessage || null,
    });
  }

  async getAiInteractionsByCase(caseId: string): Promise<any[]> {
    return await db
      .select()
      .from(aiInteractions)
      .where(eq(aiInteractions.caseId, caseId))
      .orderBy(aiInteractions.createdAt);
  }

  async getAllAiInteractions(): Promise<any[]> {
    return await db
      .select()
      .from(aiInteractions)
      .orderBy(desc(aiInteractions.createdAt));
  }

  // Wound type operations
  async getAllWoundTypes(): Promise<WoundType[]> {
    return await db
      .select()
      .from(woundTypes)
      .orderBy(desc(woundTypes.priority), woundTypes.displayName);
  }

  async getEnabledWoundTypes(): Promise<WoundType[]> {
    return await db
      .select()
      .from(woundTypes)
      .where(eq(woundTypes.isEnabled, true))
      .orderBy(desc(woundTypes.priority), woundTypes.displayName);
  }

  async getWoundType(id: number): Promise<WoundType | undefined> {
    const [woundType] = await db.select().from(woundTypes).where(eq(woundTypes.id, id));
    return woundType || undefined;
  }

  async getWoundTypeByName(name: string): Promise<WoundType | undefined> {
    // First try exact match
    let [woundType] = await db.select().from(woundTypes).where(eq(woundTypes.name, name));
    
    if (woundType) {
      return woundType;
    }
    
    // If no exact match, try with spaces converted to underscores
    const nameWithUnderscores = name.replace(/\s+/g, '_').toLowerCase();
    [woundType] = await db.select().from(woundTypes).where(eq(woundTypes.name, nameWithUnderscores));
    
    if (woundType) {
      return woundType;
    }
    
    // If still no match, try with underscores converted to spaces
    const nameWithSpaces = name.replace(/_/g, ' ').toLowerCase();
    [woundType] = await db.select().from(woundTypes).where(eq(woundTypes.name, nameWithSpaces));
    
    return woundType || undefined;
  }

  async createWoundType(woundType: InsertWoundType): Promise<WoundType> {
    const [newWoundType] = await db.insert(woundTypes).values(woundType).returning();
    return newWoundType;
  }

  async updateWoundType(id: number, updates: Partial<WoundType>): Promise<WoundType> {
    const [updated] = await db.update(woundTypes)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(woundTypes.id, id))
      .returning();
    return updated;
  }

  async toggleWoundType(id: number, enabled: boolean): Promise<WoundType> {
    const [updated] = await db.update(woundTypes)
      .set({ isEnabled: enabled, updatedAt: new Date() })
      .where(eq(woundTypes.id, id))
      .returning();
    return updated;
  }

  async deleteWoundType(id: number): Promise<boolean> {
    const [deleted] = await db.delete(woundTypes).where(eq(woundTypes.id, id)).returning();
    return !!deleted;
  }

  async getDefaultWoundType(): Promise<WoundType | undefined> {
    const [woundType] = await db.select().from(woundTypes).where(eq(woundTypes.isDefault, true));
    return woundType || undefined;
  }

  async setDefaultWoundType(id: number): Promise<WoundType> {
    // First, unset all existing defaults
    await db.update(woundTypes).set({ isDefault: false });
    
    // Then set the new default
    const [updated] = await db.update(woundTypes)
      .set({ isDefault: true, updatedAt: new Date() })
      .where(eq(woundTypes.id, id))
      .returning();
    return updated;
  }
}

export const storage = new DatabaseStorage();