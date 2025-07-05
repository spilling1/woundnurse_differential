import type { Express } from "express";
import { createServer, type Server } from "http";
import { setupAuth } from "../replitAuth";
import { registerAuthRoutes } from "./auth-routes";
import { registerAssessmentRoutes } from "./assessment-routes";
import { registerFollowUpRoutes } from "./follow-up-routes";
import { registerAdminRoutes } from "./admin-routes";

export async function registerRoutes(app: Express): Promise<Server> {
  // Auth middleware
  await setupAuth(app);

  // YOLO service proxy routes (accessible through port 5000)
  app.get('/api/yolo', async (req, res) => {
    try {
      const response = await fetch('http://localhost:8081/');
      const data = await response.json();
      res.json(data);
    } catch (error) {
      res.status(503).json({ error: 'YOLO service not available' });
    }
  });

  app.get('/api/yolo/health', async (req, res) => {
    try {
      const response = await fetch('http://localhost:8081/health');
      const data = await response.json();
      res.json(data);
    } catch (error) {
      res.status(503).json({ error: 'YOLO service not available' });
    }
  });

  // Register all route modules
  registerAuthRoutes(app);
  registerAssessmentRoutes(app);
  registerFollowUpRoutes(app);
  registerAdminRoutes(app);

  const httpServer = createServer(app);
  return httpServer;
} 