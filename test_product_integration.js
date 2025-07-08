// Test script to verify product integration
import { storage } from './server/storage.js';

async function testProductIntegration() {
  console.log('Testing product integration...');
  
  try {
    // Test 1: Get all products
    const allProducts = await storage.getAllProductRecommendations();
    console.log(`✓ Found ${allProducts.length} total products`);
    
    // Test 2: Get active products
    const activeProducts = await storage.getActiveProductRecommendations();
    console.log(`✓ Found ${activeProducts.length} active products`);
    
    // Test 3: Get products by wound type
    const diabeticProducts = await storage.getProductRecommendationsByWoundType('diabetic');
    console.log(`✓ Found ${diabeticProducts.length} diabetic wound products`);
    
    // Test 4: Get products by category
    const dressingProducts = await storage.getProductRecommendationsByCategory('wound_dressing');
    console.log(`✓ Found ${dressingProducts.length} wound dressing products`);
    
    // Test 5: Show sample products
    console.log('\nSample products:');
    activeProducts.slice(0, 3).forEach(product => {
      console.log(`- ${product.name} (${product.category})`);
      console.log(`  URL: ${product.amazonUrl}`);
      console.log(`  Wound types: ${product.woundTypes.join(', ')}`);
      console.log(`  Usage count: ${product.usageCount || 0}`);
      console.log('');
    });
    
    console.log('✓ Product integration test completed successfully!');
    
  } catch (error) {
    console.error('❌ Product integration test failed:', error);
  }
}

testProductIntegration();