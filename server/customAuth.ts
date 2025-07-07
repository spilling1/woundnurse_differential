import jwt from 'jsonwebtoken';
import bcrypt from 'bcrypt';
import type { Express, Request, Response, NextFunction, RequestHandler } from 'express';
import { storage } from './storage';
import { z } from 'zod';

// JWT secret - use SESSION_SECRET or generate a new one
const JWT_SECRET = process.env.SESSION_SECRET || 'your-jwt-secret-key';

// Permanent admin users - these users always have admin privileges
const PERMANENT_ADMINS = [
  'wardkevinpaul@gmail.com',
  'sampilling@higharc.com',
  'spilling@gmail.com'
];

// Extend Request interface to include user
declare global {
  namespace Express {
    interface Request {
      customUser?: {
        id: string;
        email: string;
        role: string;
        mustChangePassword?: boolean;
      };
    }
  }
}

// Validation schemas
const loginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(1),
});

const registerSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
  firstName: z.string().min(1),
  lastName: z.string().min(1),
});

const changePasswordSchema = z.object({
  currentPassword: z.string().min(1),
  newPassword: z.string().min(8),
});

// Generate JWT token
export function generateToken(user: { id: string; email: string; role: string; mustChangePassword?: boolean }) {
  // Check if user is a permanent admin
  const isPermanentAdmin = PERMANENT_ADMINS.includes(user.email || '');
  const userRole = isPermanentAdmin ? 'admin' : user.role;
  
  return jwt.sign(
    { 
      id: user.id, 
      email: user.email, 
      role: userRole,
      mustChangePassword: user.mustChangePassword 
    },
    JWT_SECRET,
    { expiresIn: '7d' }
  );
}

// Verify JWT token
export function verifyToken(token: string) {
  try {
    return jwt.verify(token, JWT_SECRET) as { 
      id: string; 
      email: string; 
      role: string; 
      mustChangePassword?: boolean;
    };
  } catch (error) {
    return null;
  }
}

// Authentication middleware
export const isAuthenticated: RequestHandler = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;
    const token = authHeader?.startsWith('Bearer ') ? authHeader.substring(7) : null;

    if (!token) {
      return res.status(401).json({ message: 'No token provided' });
    }

    const decoded = verifyToken(token);
    if (!decoded) {
      return res.status(401).json({ message: 'Invalid token' });
    }

    // Get fresh user data from database
    const user = await storage.getUser(decoded.id);
    if (!user || user.status !== 'active') {
      return res.status(401).json({ message: 'User not found or inactive' });
    }

    // Check if user is a permanent admin
    const isPermanentAdmin = PERMANENT_ADMINS.includes(user.email || '');
    const userRole = isPermanentAdmin ? 'admin' : user.role;

    req.customUser = {
      id: user.id,
      email: user.email!,
      role: userRole,
      mustChangePassword: user.mustChangePassword || false,
    };

    next();
  } catch (error) {
    console.error('Authentication error:', error);
    res.status(401).json({ message: 'Authentication failed' });
  }
};

// Admin middleware
export const isAdmin: RequestHandler = async (req, res, next) => {
  if (!req.customUser || req.customUser.role !== 'admin') {
    return res.status(403).json({ message: 'Admin access required' });
  }
  next();
};

// Setup authentication routes
export function setupCustomAuth(app: Express) {
  // Login endpoint
  app.post('/api/auth/login', async (req, res) => {
    try {
      const { email, password } = loginSchema.parse(req.body);

      // Find user by email
      const user = await storage.getUserByEmail(email);
      if (!user) {
        return res.status(401).json({ message: 'Invalid credentials' });
      }

      // Check password
      if (!user.password) {
        return res.status(401).json({ message: 'Password not set for this user' });
      }

      const isValid = await bcrypt.compare(password, user.password);
      if (!isValid) {
        return res.status(401).json({ message: 'Invalid credentials' });
      }

      // Update last login
      await storage.updateUser(user.id, { lastLoginAt: new Date() });

      // Generate token
      const token = generateToken({
        id: user.id,
        email: user.email!,
        role: user.role,
        mustChangePassword: user.mustChangePassword || false,
      });

      res.json({
        token,
        user: {
          id: user.id,
          email: user.email,
          firstName: user.firstName,
          lastName: user.lastName,
          role: user.role,
          mustChangePassword: user.mustChangePassword || false,
        },
      });
    } catch (error) {
      console.error('Login error:', error);
      res.status(400).json({ message: 'Login failed' });
    }
  });

  // Register endpoint
  app.post('/api/auth/register', async (req, res) => {
    try {
      const { email, password, firstName, lastName } = registerSchema.parse(req.body);

      // Check if user already exists
      const existingUser = await storage.getUserByEmail(email);
      if (existingUser) {
        return res.status(400).json({ message: 'User already exists' });
      }

      // Hash password
      const hashedPassword = await bcrypt.hash(password, 10);

      // Generate unique ID
      const userId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      // Create user
      const user = await storage.upsertUser({
        id: userId,
        email,
        password: hashedPassword,
        firstName,
        lastName,
        role: 'user',
        status: 'active',
        mustChangePassword: false,
      });

      // Generate token
      const token = generateToken({
        id: user.id,
        email: user.email!,
        role: user.role,
        mustChangePassword: false,
      });

      res.status(201).json({
        token,
        user: {
          id: user.id,
          email: user.email,
          firstName: user.firstName,
          lastName: user.lastName,
          role: user.role,
          mustChangePassword: false,
        },
      });
    } catch (error) {
      console.error('Registration error:', error);
      res.status(400).json({ message: 'Registration failed' });
    }
  });

  // Change password endpoint
  app.post('/api/auth/change-password', isAuthenticated, async (req, res) => {
    try {
      const { currentPassword, newPassword } = changePasswordSchema.parse(req.body);

      // Get user
      const user = await storage.getUser(req.customUser!.id);
      if (!user || !user.password) {
        return res.status(400).json({ message: 'User not found' });
      }

      // Verify current password
      const isValid = await bcrypt.compare(currentPassword, user.password);
      if (!isValid) {
        return res.status(401).json({ message: 'Current password is incorrect' });
      }

      // Hash new password
      const hashedPassword = await bcrypt.hash(newPassword, 10);

      // Update user
      await storage.updateUser(user.id, { 
        password: hashedPassword,
        mustChangePassword: false,
      });

      res.json({ message: 'Password changed successfully' });
    } catch (error) {
      console.error('Change password error:', error);
      res.status(400).json({ message: 'Failed to change password' });
    }
  });

  // Get current user endpoint
  app.get('/api/auth/user', isAuthenticated, async (req, res) => {
    try {
      const user = await storage.getUser(req.customUser!.id);
      if (!user) {
        return res.status(404).json({ message: 'User not found' });
      }

      res.json({
        id: user.id,
        email: user.email,
        firstName: user.firstName,
        lastName: user.lastName,
        role: user.role,
        mustChangePassword: user.mustChangePassword || false,
      });
    } catch (error) {
      console.error('Get user error:', error);
      res.status(500).json({ message: 'Failed to fetch user' });
    }
  });

  // Logout endpoint (client-side token removal)
  app.post('/api/auth/logout', (req, res) => {
    res.json({ message: 'Logged out successfully' });
  });
}