import type { Express } from "express";
import { storage } from "../storage";
import { isAuthenticated } from "../customAuth";
import { insertUserProfileSchema } from "@shared/schema";
import { z } from "zod";

export function registerProfileRoutes(app: Express) {
  // Get user profile
  app.get('/api/profile', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.customUser.id;
      const profile = await storage.getUserProfile(userId);
      res.json(profile);
    } catch (error) {
      console.error("Error fetching user profile:", error);
      res.status(500).json({ message: "Failed to fetch profile" });
    }
  });

  // Create or update user profile
  app.post('/api/profile', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.customUser.id;
      
      // Validate request body
      const profileData = insertUserProfileSchema.parse({
        ...req.body,
        userId: userId
      });

      // Check if profile already exists
      const existingProfile = await storage.getUserProfile(userId);
      
      let profile;
      if (existingProfile) {
        // Update existing profile
        profile = await storage.updateUserProfile(userId, profileData);
      } else {
        // Create new profile
        profile = await storage.createUserProfile(profileData);
      }

      res.json(profile);
    } catch (error) {
      console.error("Error saving user profile:", error);
      if (error instanceof z.ZodError) {
        res.status(400).json({ message: "Invalid profile data", errors: error.errors });
      } else {
        res.status(500).json({ message: "Failed to save profile" });
      }
    }
  });

  // Update user profile
  app.patch('/api/profile', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.customUser.id;
      
      // Validate request body (partial update)
      const profileData = insertUserProfileSchema.partial().parse(req.body);

      const profile = await storage.updateUserProfile(userId, profileData);
      res.json(profile);
    } catch (error) {
      console.error("Error updating user profile:", error);
      if (error instanceof z.ZodError) {
        res.status(400).json({ message: "Invalid profile data", errors: error.errors });
      } else {
        res.status(500).json({ message: "Failed to update profile" });
      }
    }
  });

  // Delete user profile
  app.delete('/api/profile', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.customUser.id;
      const success = await storage.deleteUserProfile(userId);
      
      if (success) {
        res.json({ message: "Profile deleted successfully" });
      } else {
        res.status(404).json({ message: "Profile not found" });
      }
    } catch (error) {
      console.error("Error deleting user profile:", error);
      res.status(500).json({ message: "Failed to delete profile" });
    }
  });
}