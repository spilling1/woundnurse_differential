import { woundAssessments, feedbacks, agentInstructions, users, type WoundAssessment, type InsertWoundAssessment, type Feedback, type InsertFeedback, type AgentInstructions, type InsertAgentInstructions, type User, type UpsertUser } from "@shared/schema";
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
  createFeedback(feedback: InsertFeedback): Promise<Feedback>;
  getFeedbacksByCase(caseId: string): Promise<Feedback[]>;
  getActiveAgentInstructions(): Promise<AgentInstructions | undefined>;
  createAgentInstructions(instructions: InsertAgentInstructions): Promise<AgentInstructions>;
  updateAgentInstructions(id: number, content: string): Promise<AgentInstructions>;
}

export class DatabaseStorage implements IStorage {
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

  async updateAgentInstructions(id: number, content: string): Promise<AgentInstructions> {
    const [result] = await db
      .update(agentInstructions)
      .set({ content, updatedAt: new Date() })
      .where(eq(agentInstructions.id, id))
      .returning();
    return result;
  }
}

export const storage = new DatabaseStorage();