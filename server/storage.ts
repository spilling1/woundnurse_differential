import { woundAssessments, feedbacks, type WoundAssessment, type InsertWoundAssessment, type Feedback, type InsertFeedback } from "@shared/schema";

export interface IStorage {
  createWoundAssessment(assessment: InsertWoundAssessment): Promise<WoundAssessment>;
  getWoundAssessment(caseId: string): Promise<WoundAssessment | undefined>;
  createFeedback(feedback: InsertFeedback): Promise<Feedback>;
  getFeedbacksByCase(caseId: string): Promise<Feedback[]>;
}

export class MemStorage implements IStorage {
  private assessments: Map<string, WoundAssessment>;
  private feedbacks: Map<number, Feedback>;
  private currentAssessmentId: number;
  private currentFeedbackId: number;

  constructor() {
    this.assessments = new Map();
    this.feedbacks = new Map();
    this.currentAssessmentId = 1;
    this.currentFeedbackId = 1;
  }

  async createWoundAssessment(insertAssessment: InsertWoundAssessment): Promise<WoundAssessment> {
    const id = this.currentAssessmentId++;
    const assessment: WoundAssessment = {
      ...insertAssessment,
      id,
      version: insertAssessment.version || "v1.0.0",
      classification: insertAssessment.classification || null,
      createdAt: new Date(),
    };
    this.assessments.set(assessment.caseId, assessment);
    return assessment;
  }

  async getWoundAssessment(caseId: string): Promise<WoundAssessment | undefined> {
    return this.assessments.get(caseId);
  }

  async createFeedback(insertFeedback: InsertFeedback): Promise<Feedback> {
    const id = this.currentFeedbackId++;
    const feedback: Feedback = {
      ...insertFeedback,
      id,
      comments: insertFeedback.comments || null,
      createdAt: new Date(),
    };
    this.feedbacks.set(id, feedback);
    return feedback;
  }

  async getFeedbacksByCase(caseId: string): Promise<Feedback[]> {
    return Array.from(this.feedbacks.values()).filter(
      (feedback) => feedback.caseId === caseId
    );
  }
}

export const storage = new MemStorage();
