import { storage } from '../storage';
import { LoggerService } from '../services/loggerService';

/**
 * Extract product recommendations from care plan HTML/text
 */
function extractProductsFromCarePlan(carePlan: string): Array<{
  category: string;
  productName: string;
  amazonLink: string;
  reason: string;
}> {
  const products: Array<{
    category: string;
    productName: string;
    amazonLink: string;
    reason: string;
  }> = [];
  
  try {
    // Look for Amazon links in the care plan
    const amazonLinkRegex = /\[([^\]]+)\]\(https:\/\/www\.amazon\.com\/s\?k=([^)]+)\)/g;
    let match;
    
    while ((match = amazonLinkRegex.exec(carePlan)) !== null) {
      const productName = match[1];
      const amazonLink = match[0].match(/\(([^)]+)\)/)?.[1] || '';
      
      // Try to determine category based on context
      let category = 'General';
      const lowerName = productName.toLowerCase();
      
      if (lowerName.includes('dressing') || lowerName.includes('bandage')) {
        category = 'Wound Dressing';
      } else if (lowerName.includes('cleanser') || lowerName.includes('saline')) {
        category = 'Wound Cleanser';
      } else if (lowerName.includes('moisturizer') || lowerName.includes('barrier')) {
        category = 'Skin Care';
      } else if (lowerName.includes('gloves') || lowerName.includes('gauze')) {
        category = 'Supplies';
      } else if (lowerName.includes('compression') || lowerName.includes('sock')) {
        category = 'Compression';
      }
      
      // Extract context/reason from surrounding text
      const contextStart = Math.max(0, match.index - 100);
      const contextEnd = Math.min(carePlan.length, match.index + match[0].length + 100);
      const context = carePlan.substring(contextStart, contextEnd);
      
      let reason = 'Recommended for wound care';
      if (context.includes('infection')) {
        reason = 'Helps prevent infection';
      } else if (context.includes('moist') || context.includes('hydrat')) {
        reason = 'Maintains moist healing environment';
      } else if (context.includes('protect')) {
        reason = 'Provides protection';
      } else if (context.includes('clean')) {
        reason = 'For wound cleaning';
      }
      
      products.push({
        category,
        productName,
        amazonLink,
        reason
      });
    }
    
    return products;
  } catch (error) {
    console.error('Error extracting products from care plan:', error);
    return [];
  }
}

async function backfillLogs() {
  try {
    console.log('Starting backfill process...');
    
    // Get all wound assessments
    const assessments = await storage.getAllWoundAssessments();
    console.log(`Found ${assessments.length} assessments to process`);
    
    // Get all AI interactions
    const interactions = await storage.getAllAiInteractions();
    console.log(`Found ${interactions.length} AI interactions to process`);
    
    let qaEntries = 0;
    let productEntries = 0;
    
    // Process Q&A from AI interactions
    const caseQAMap = new Map();
    
    for (const interaction of interactions) {
      // Look for user_question_responses interactions (when users answer questions)
      if (interaction.stepType === 'user_question_responses' && interaction.parsedResult?.questions) {
        const questions = interaction.parsedResult.questions;
        const answeredQuestions = questions.filter((q: any) => q.answer && q.answer.trim() !== '');
        
        if (answeredQuestions.length > 0) {
          if (!caseQAMap.has(interaction.caseId)) {
            caseQAMap.set(interaction.caseId, {
              questions: [],
              reassessment: interaction.parsedResult.reassessment || 'No reassessment available'
            });
          }
          
          caseQAMap.get(interaction.caseId).questions.push(...answeredQuestions);
        }
      }
      
      // Also look for question_generation interactions with answered questions
      if (interaction.stepType === 'question_generation' && interaction.parsedResult?.questions) {
        const questions = interaction.parsedResult.questions;
        const answeredQuestions = questions.filter((q: any) => q.answer && q.answer.trim() !== '');
        
        if (answeredQuestions.length > 0) {
          if (!caseQAMap.has(interaction.caseId)) {
            caseQAMap.set(interaction.caseId, {
              questions: [],
              reassessment: interaction.parsedResult.reassessment || 'No reassessment available'
            });
          }
          
          caseQAMap.get(interaction.caseId).questions.push(...answeredQuestions);
        }
      }
    }
    
    // Process assessments for Q&A and products
    for (const assessment of assessments) {
      console.log(`Processing case: ${assessment.caseId}`);
      
      // Process Q&A from AI interactions
      const qaData = caseQAMap.get(assessment.caseId);
      if (qaData && qaData.questions.length > 0) {
        const user = await storage.getUser(assessment.userId);
        
        await LoggerService.logQAInteraction({
          caseId: assessment.caseId,
          userEmail: user?.email || 'unknown',
          timestamp: assessment.createdAt,
          woundType: assessment.classification?.woundType || 'unknown',
          audience: assessment.audience,
          aiModel: assessment.model,
          questions: qaData.questions.map((q: any) => ({
            question: q.question,
            answer: q.answer || '',
            category: q.category || 'unknown',
            confidenceImpact: q.confidenceImpact || 'unknown'
          })),
          finalConfidence: assessment.classification?.confidence || 0,
          reassessment: qaData.reassessment
        });
        
        qaEntries++;
        console.log(`  ✓ Added Q&A entry for case ${assessment.caseId} (${qaData.questions.length} questions)`);
      }
      
      // Also check traditional question storage
      const questions = await storage.getQuestionsBySession(assessment.caseId);
      if (questions && questions.length > 0 && !caseQAMap.has(assessment.caseId)) {
        const answeredQuestions = questions.filter(q => q.answer && q.answer.trim() !== '');
        
        if (answeredQuestions.length > 0) {
          const user = await storage.getUser(assessment.userId);
          
          await LoggerService.logQAInteraction({
            caseId: assessment.caseId,
            userEmail: user?.email || 'unknown',
            timestamp: assessment.createdAt,
            woundType: assessment.classification?.woundType || 'unknown',
            audience: assessment.audience,
            aiModel: assessment.model,
            questions: answeredQuestions.map(q => ({
              question: q.question,
              answer: q.answer || '',
              category: q.category || 'unknown',
              confidenceImpact: q.confidenceImpact || 'unknown'
            })),
            finalConfidence: assessment.classification?.confidence || 0,
            reassessment: 'Historical data - no reassessment recorded'
          });
          
          qaEntries++;
          console.log(`  ✓ Added Q&A entry for case ${assessment.caseId} (from traditional storage)`);
        }
      }
      
      // Process product recommendations from care plan
      if (assessment.carePlan) {
        const products = extractProductsFromCarePlan(assessment.carePlan);
        
        if (products.length > 0) {
          const user = await storage.getUser(assessment.userId);
          
          await LoggerService.logProductRecommendations({
            caseId: assessment.caseId,
            userEmail: user?.email || 'unknown',
            timestamp: assessment.createdAt,
            woundType: assessment.classification?.woundType || 'unknown',
            audience: assessment.audience,
            aiModel: assessment.model,
            products: products
          });
          
          productEntries++;
          console.log(`  ✓ Added ${products.length} product recommendations for case ${assessment.caseId}`);
        }
      }
    }
    
    console.log('\n=== Backfill Complete ===');
    console.log(`Q&A entries added: ${qaEntries}`);
    console.log(`Product recommendation entries added: ${productEntries}`);
    console.log(`Total assessments processed: ${assessments.length}`);
    
  } catch (error) {
    console.error('Error during backfill:', error);
  }
}

// Run the backfill
backfillLogs().then(() => {
  console.log('Backfill process completed');
  process.exit(0);
}).catch(error => {
  console.error('Backfill failed:', error);
  process.exit(1);
});