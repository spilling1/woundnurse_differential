import { woundAssessments, feedbacks, agentInstructions, agentQuestions, users, type WoundAssessment, type InsertWoundAssessment, type Feedback, type InsertFeedback, type AgentInstructions, type InsertAgentInstructions, type AgentQuestion, type InsertAgentQuestion, type User, type UpsertUser } from "@shared/schema";
import { db } from "./db";
import { eq, desc } from "drizzle-orm";

export interface IStorage {
  // User operations - required for Replit Auth
  getUser(id: string): Promise<User | undefined>;
  upsertUser(user: UpsertUser): Promise<User>;
  getUserWoundAssessments(userId: string): Promise<WoundAssessment[]>;
  
  // Wound assessment operations
  createWoundAssessment(assessment: InsertWoundAssessment): Promise<WoundAssessment>;
  getWoundAssessment(caseId: string): Promise<WoundAssessment | undefined>;
  getLatestWoundAssessment(caseId: string): Promise<WoundAssessment | undefined>;
  getWoundAssessmentHistory(caseId: string): Promise<WoundAssessment[]>;
  createFollowUpAssessment(assessment: InsertWoundAssessment): Promise<WoundAssessment>;
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
  }): Promise<AgentInstructions>;
  
  // Agent question operations
  createAgentQuestion(question: InsertAgentQuestion): Promise<AgentQuestion>;
  getQuestionsBySession(sessionId: string): Promise<AgentQuestion[]>;
  answerQuestion(questionId: number, answer: string): Promise<AgentQuestion>;
  getUnansweredQuestions(sessionId: string): Promise<AgentQuestion[]>;
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
}

export const storage = new DatabaseStorage();