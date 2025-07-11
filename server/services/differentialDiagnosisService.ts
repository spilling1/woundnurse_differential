import { callOpenAI } from './openai';
import { callGemini } from './gemini';
import { storage } from '../storage';

export interface DifferentialDiagnosisQuestion {
  id: string;
  question: string;
  category: 'medical_history' | 'clinical_signs' | 'diagnostic_tests' | 'timeline';
  importance: 'critical' | 'high' | 'medium';
  order: number;
}

export interface DiagnosisRefinement {
  originalDiagnosis: any;
  refinedDiagnosis: any;
  eliminatedPossibilities: string[];
  remainingPossibilities: any[];
  confidence: number;
  reasoning: string;
  questionsAnalyzed: any[];
}

class DifferentialDiagnosisService {
  
  /**
   * Normalize probabilities to add up to 100%
   */
  normalizeProbabilities(possibleTypes: any[]): any[] {
    const totalConfidence = possibleTypes.reduce((sum, type) => sum + parseFloat(type.confidence), 0);
    
    if (totalConfidence === 0) return possibleTypes;
    
    return possibleTypes.map(type => ({
      ...type,
      confidence: parseFloat(type.confidence) / totalConfidence
    }));
  }

  /**
   * Create interactive differential diagnosis questions ordered by clinical importance
   */
  createOrderedQuestions(differentialDiagnosis: any): DifferentialDiagnosisQuestion[] {
    const questions: DifferentialDiagnosisQuestion[] = [];
    
    if (!differentialDiagnosis?.questionsToAsk) return questions;
    
    // Order questions by clinical importance for differential diagnosis
    const questionMapping = [
      {
        keywords: ['diabetes', 'diabetic', 'blood sugar', 'glucose'],
        category: 'medical_history',
        importance: 'critical',
        order: 1
      },
      {
        keywords: ['mobility', 'bedridden', 'wheelchair', 'movement'],
        category: 'clinical_signs',
        importance: 'critical',
        order: 2
      },
      {
        keywords: ['infection', 'fever', 'chills', 'systemic'],
        category: 'clinical_signs',
        importance: 'high',
        order: 3
      },
      {
        keywords: ['pain', 'numb', 'sensation', 'feel'],
        category: 'clinical_signs',
        importance: 'high',
        order: 4
      },
      {
        keywords: ['origin', 'start', 'begin', 'caused', 'how'],
        category: 'timeline',
        importance: 'high',
        order: 5
      }
    ];
    
    differentialDiagnosis.questionsToAsk.forEach((question: string, index: number) => {
      const mapping = questionMapping.find(m => 
        m.keywords.some(keyword => question.toLowerCase().includes(keyword))
      );
      
      questions.push({
        id: `dd_q${index + 1}`,
        question,
        category: mapping?.category || 'clinical_signs',
        importance: mapping?.importance || 'medium',
        order: mapping?.order || (index + 6)
      });
    });
    
    // Sort by order (clinical importance)
    return questions.sort((a, b) => a.order - b.order);
  }

  /**
   * Refine differential diagnosis based on user answers
   */
  async refineDifferentialDiagnosis(
    originalClassification: any,
    questionAnswers: any[],
    model: string = 'gemini-2.5-pro'
  ): Promise<DiagnosisRefinement> {
    
    const prompt = this.createRefinementPrompt(originalClassification, questionAnswers);
    
    try {
      let response;
      
      if (model.startsWith('gemini')) {
        response = await callGemini(model, prompt);
      } else {
        response = await callOpenAI(model, prompt);
      }
      
      const refinedData = this.parseRefinementResponse(response);
      
      // Create refined diagnosis with complete wound classification structure
      const refinedDiagnosis = {
        ...originalClassification,
        woundType: refinedData.diagnosis.primaryDiagnosis,
        confidence: refinedData.confidence,
        differentialDiagnosis: {
          ...originalClassification.differentialDiagnosis,
          possibleTypes: this.normalizeProbabilities(refinedData.remaining)
        }
      };

      return {
        originalDiagnosis: originalClassification,
        refinedDiagnosis: refinedDiagnosis,
        eliminatedPossibilities: refinedData.eliminated,
        remainingPossibilities: this.normalizeProbabilities(refinedData.remaining),
        confidence: refinedData.confidence,
        reasoning: refinedData.reasoning,
        questionsAnalyzed: questionAnswers
      };
      
    } catch (error) {
      console.error('DifferentialDiagnosisService: Error refining diagnosis:', error);
      
      // Fallback refinement based on simple logic
      return this.createFallbackRefinement(originalClassification, questionAnswers);
    }
  }

  /**
   * Create refinement prompt for AI analysis
   */
  private createRefinementPrompt(originalClassification: any, questionAnswers: any[]): string {
    const answersText = questionAnswers.map(qa => 
      `Q: ${qa.question}\nA: ${qa.answer}`
    ).join('\n\n');
    
    return `You are analyzing wound differential diagnosis refinement based on clinical answers.

ORIGINAL ASSESSMENT:
${JSON.stringify(originalClassification.differentialDiagnosis, null, 2)}

PATIENT ANSWERS:
${answersText}

CRITICAL INSTRUCTIONS:
1. Analyze how each answer affects the probability of each differential diagnosis
2. ELIMINATE possibilities that are ruled out by the answers
3. INCREASE probability for diagnoses supported by the answers
4. NORMALIZE final probabilities to add up to 100%
5. Provide clear clinical reasoning for each change

REQUIRED RESPONSE FORMAT (JSON):
{
  "diagnosis": {
    "primaryDiagnosis": "Most likely diagnosis name",
    "confidence": 0.85,
    "stage": "if applicable",
    "reasoning": "Clinical reasoning for final diagnosis"
  },
  "eliminated": ["Diagnosis 1", "Diagnosis 2"],
  "remaining": [
    {
      "woundType": "Primary Diagnosis",
      "confidence": 0.70,
      "reasoning": "Why this is most likely based on answers"
    },
    {
      "woundType": "Secondary Diagnosis", 
      "confidence": 0.30,
      "reasoning": "Why this is still possible"
    }
  ],
  "confidence": 0.85,
  "reasoning": "Overall clinical reasoning based on patient answers. Explain how each answer influenced the diagnostic probabilities."
}

IMPORTANT: 
- Probabilities in "remaining" MUST add up to 1.0 (100%)
- Use evidence-based clinical reasoning
- Address each patient answer explicitly
- If diabetes is confirmed, prioritize diabetic foot ulcer
- If mobility is normal, reduce pressure ulcer probability
- If infection signs are present, adjust accordingly`;
  }

  /**
   * Parse AI response for refinement data
   */
  private parseRefinementResponse(response: string): any {
    try {
      // Extract JSON from response
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (!jsonMatch) throw new Error('No JSON found in response');
      
      const parsed = JSON.parse(jsonMatch[0]);
      
      // Validate and normalize probabilities
      if (parsed.remaining && Array.isArray(parsed.remaining)) {
        parsed.remaining = this.normalizeProbabilities(parsed.remaining);
      }
      
      return parsed;
      
    } catch (error) {
      console.error('DifferentialDiagnosisService: Error parsing response:', error);
      throw error;
    }
  }

  /**
   * Create fallback refinement when AI fails
   */
  private createFallbackRefinement(originalClassification: any, questionAnswers: any[]): DiagnosisRefinement {
    const originalTypes = originalClassification.differentialDiagnosis?.possibleTypes || [];
    
    // Simple logic-based refinement
    let remaining = [...originalTypes];
    let eliminated: string[] = [];
    
    // Check for diabetes answer
    const diabetesAnswer = questionAnswers.find(qa => 
      qa.question.toLowerCase().includes('diabetes')
    );
    
    if (diabetesAnswer?.answer.toLowerCase().includes('yes')) {
      // Boost diabetic ulcer probability
      remaining = remaining.map(type => ({
        ...type,
        confidence: type.woundType.toLowerCase().includes('diabetic') ? 
          Math.min(0.8, parseFloat(type.confidence) * 1.5) : 
          parseFloat(type.confidence) * 0.8
      }));
    }
    
    // Check for mobility answer
    const mobilityAnswer = questionAnswers.find(qa => 
      qa.question.toLowerCase().includes('mobility') || 
      qa.question.toLowerCase().includes('bedridden')
    );
    
    if (mobilityAnswer?.answer.toLowerCase().includes('no')) {
      // Reduce pressure ulcer probability
      remaining = remaining.map(type => ({
        ...type,
        confidence: type.woundType.toLowerCase().includes('pressure') ? 
          parseFloat(type.confidence) * 0.3 : 
          parseFloat(type.confidence) * 1.2
      }));
    }
    
    // Normalize probabilities
    remaining = this.normalizeProbabilities(remaining);
    
    // Filter out very low probability diagnoses (15% or below)
    const threshold = 0.15;
    eliminated = remaining.filter(type => type.confidence <= threshold).map(type => type.woundType);
    remaining = remaining.filter(type => type.confidence > threshold);
    
    // Create refined diagnosis with complete wound classification structure
    const refinedDiagnosis = {
      ...originalClassification,
      woundType: remaining[0]?.woundType || 'Uncertain',
      confidence: remaining[0]?.confidence || 0.5,
      differentialDiagnosis: {
        ...originalClassification.differentialDiagnosis,
        possibleTypes: remaining
      }
    };

    return {
      originalDiagnosis: originalClassification,
      refinedDiagnosis: refinedDiagnosis,
      eliminatedPossibilities: eliminated,
      remainingPossibilities: remaining,
      confidence: remaining[0]?.confidence || 0.5,
      reasoning: 'Analysis based on patient answers with clinical logic-based refinement',
      questionsAnalyzed: questionAnswers
    };
  }
}

export const differentialDiagnosisService = new DifferentialDiagnosisService();