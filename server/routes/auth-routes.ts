import type { Express } from "express";
import { storage } from "../storage";
import { isAuthenticated } from "../replitAuth";

export function registerAuthRoutes(app: Express): void {
  // Get authenticated user
  app.get('/api/auth/user', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const user = await storage.getUser(userId);
      res.json(user);
    } catch (error) {
      console.error("Error fetching user:", error);
      res.status(500).json({ message: "Failed to fetch user" });
    }
  });

  // Get user's wound assessments
  app.get('/api/my-cases', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const assessments = await storage.getUserWoundAssessments(userId);
      res.json(assessments);
    } catch (error) {
      console.error("Error fetching user cases:", error);
      res.status(500).json({ message: "Failed to fetch cases" });
    }
  });
} 