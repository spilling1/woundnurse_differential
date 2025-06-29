import { woundAssessments, feedbacks, agentInstructions, type WoundAssessment, type InsertWoundAssessment, type Feedback, type InsertFeedback, type AgentInstructions, type InsertAgentInstructions } from "@shared/schema";
import { db } from "./db";
import { eq, desc } from "drizzle-orm";

export interface IStorage {
  createWoundAssessment(assessment: InsertWoundAssessment): Promise<WoundAssessment>;
  getWoundAssessment(caseId: string): Promise<WoundAssessment | undefined>;
  createFeedback(feedback: InsertFeedback): Promise<Feedback>;
  getFeedbacksByCase(caseId: string): Promise<Feedback[]>;
  getActiveAgentInstructions(): Promise<AgentInstructions | undefined>;
  createAgentInstructions(instructions: InsertAgentInstructions): Promise<AgentInstructions>;
  updateAgentInstructions(id: number, content: string): Promise<AgentInstructions>;
}

export class DatabaseStorage implements IStorage {
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