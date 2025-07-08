import express from 'express';
import { storage } from '../storage';
import { insertProductRecommendationSchema } from '@shared/schema';
import { z } from 'zod';
import { isAuthenticated } from '../customAuth';

const router = express.Router();

// Apply authentication middleware to all routes
router.use(isAuthenticated);

// Get all product recommendations (admin only)
router.get('/all', async (req, res) => {
  try {
    const customUser = (req as any).customUser;
    if (!customUser?.isAdmin) {
      return res.status(403).json({ error: 'Admin access required' });
    }

    const products = await storage.getAllProductRecommendations();
    res.json(products);
  } catch (error) {
    console.error('Error getting all product recommendations:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get active product recommendations
router.get('/active', async (req, res) => {
  try {
    const products = await storage.getActiveProductRecommendations();
    res.json(products);
  } catch (error) {
    console.error('Error getting active product recommendations:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get product recommendations by category
router.get('/category/:category', async (req, res) => {
  try {
    const { category } = req.params;
    const products = await storage.getProductRecommendationsByCategory(category);
    res.json(products);
  } catch (error) {
    console.error('Error getting product recommendations by category:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get product recommendations by wound type
router.get('/wound-type/:woundType', async (req, res) => {
  try {
    const { woundType } = req.params;
    const products = await storage.getProductRecommendationsByWoundType(woundType);
    res.json(products);
  } catch (error) {
    console.error('Error getting product recommendations by wound type:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get single product recommendation
router.get('/:id', async (req, res) => {
  try {
    const customUser = (req as any).customUser;
    if (!customUser?.isAdmin) {
      return res.status(403).json({ error: 'Admin access required' });
    }

    const id = parseInt(req.params.id);
    const product = await storage.getProductRecommendation(id);
    
    if (!product) {
      return res.status(404).json({ error: 'Product not found' });
    }

    res.json(product);
  } catch (error) {
    console.error('Error getting product recommendation:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Create new product recommendation (admin only)
router.post('/', async (req, res) => {
  try {
    const customUser = (req as any).customUser;
    if (!customUser?.isAdmin) {
      return res.status(403).json({ error: 'Admin access required' });
    }

    const validatedData = insertProductRecommendationSchema.parse(req.body);
    const product = await storage.createProductRecommendation(validatedData);
    res.status(201).json(product);
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: 'Validation error', details: error.errors });
    }
    console.error('Error creating product recommendation:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Update product recommendation (admin only)
router.put('/:id', async (req, res) => {
  try {
    const customUser = (req as any).customUser;
    if (!customUser?.isAdmin) {
      return res.status(403).json({ error: 'Admin access required' });
    }

    const id = parseInt(req.params.id);
    const updates = req.body;
    
    const product = await storage.updateProductRecommendation(id, updates);
    res.json(product);
  } catch (error) {
    console.error('Error updating product recommendation:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Toggle product recommendation active status (admin only)
router.patch('/:id/toggle', async (req, res) => {
  try {
    const customUser = (req as any).customUser;
    if (!customUser?.isAdmin) {
      return res.status(403).json({ error: 'Admin access required' });
    }

    const id = parseInt(req.params.id);
    const { isActive } = req.body;
    
    const product = await storage.toggleProductRecommendation(id, isActive);
    res.json(product);
  } catch (error) {
    console.error('Error toggling product recommendation:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Increment product usage count (internal use)
router.post('/:id/increment-usage', async (req, res) => {
  try {
    const id = parseInt(req.params.id);
    const product = await storage.incrementProductRecommendationUsage(id);
    res.json(product);
  } catch (error) {
    console.error('Error incrementing product usage:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Delete product recommendation (admin only)
router.delete('/:id', async (req, res) => {
  try {
    const customUser = (req as any).customUser;
    if (!customUser?.isAdmin) {
      return res.status(403).json({ error: 'Admin access required' });
    }

    const id = parseInt(req.params.id);
    const deleted = await storage.deleteProductRecommendation(id);
    
    if (!deleted) {
      return res.status(404).json({ error: 'Product not found' });
    }

    res.json({ message: 'Product recommendation deleted successfully' });
  } catch (error) {
    console.error('Error deleting product recommendation:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

export default router;