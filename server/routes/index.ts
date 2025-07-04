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

  // Register all route modules
  registerAuthRoutes(app);
  registerAssessmentRoutes(app);
  registerFollowUpRoutes(app);
  registerAdminRoutes(app);

  const httpServer = createServer(app);
  return httpServer;
} 