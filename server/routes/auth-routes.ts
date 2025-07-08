import type { Express } from "express";
import { storage } from "../storage";
import { isAuthenticated } from "../customAuth";

export function registerAuthRoutes(app: Express): void {
  // Note: /api/auth/user endpoint is now handled in customAuth.ts
  
  // Logout endpoint (for compatibility with old system)
  app.get('/api/logout', (req, res) => {
    res.json({ message: "Logged out successfully" });
  });
  
  // Get user's wound assessments
  app.get('/api/my-cases', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.customUser.id;
      let assessments = await storage.getUserWoundAssessments(userId);
      
      // Also get assessments without userId (legacy assessments) and assign them to current user
      const allAssessments = await storage.getAllWoundAssessments();
      const orphanedAssessments = allAssessments.filter(a => !a.userId);
      
      // Update orphaned assessments to belong to current user
      for (const assessment of orphanedAssessments) {
        try {
          await storage.updateWoundAssessment(assessment.caseId, assessment.versionNumber, {
            userId: userId
          });
          assessment.userId = userId;
          assessments.push(assessment);
        } catch (error) {
          console.error('Error updating orphaned assessment:', error);
        }
      }
      
      res.json(assessments);
    } catch (error) {
      console.error("Error fetching user cases:", error);
      res.status(500).json({ message: "Failed to fetch cases" });
    }
  });

  // Update case name
  app.patch('/api/case/:caseId/name', isAuthenticated, async (req: any, res) => {
    try {
      const { caseId } = req.params;
      const { caseName } = req.body;
      
      if (!caseName || typeof caseName !== 'string' || caseName.trim().length === 0) {
        return res.status(400).json({ message: "Case name is required" });
      }

      await storage.updateCaseName(caseId, caseName.trim());
      res.json({ message: "Case name updated successfully" });
    } catch (error) {
      console.error("Error updating case name:", error);
      res.status(500).json({ message: "Failed to update case name" });
    }
  });
} 