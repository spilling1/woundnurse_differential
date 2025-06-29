import { callOpenAI } from './openai';
import { callGemini } from './gemini';
import { storage } from '../storage';
import { InsertAgentQuestion } from '@shared/schema';

interface QuestionAnalysisResult {
  needsQuestions: boolean;
  questions: string[];
  questionTypes: string[];
  reasoning: string;
}

export async function analyzeAssessmentForQuestions(
  imageBase64: string,
  contextData: any,
  audience: string,
  model: string,
  userId: string,
  sessionId: string
): Promise<QuestionAnalysisResult> {
  const analysisPrompt = `
You are an AI wound care specialist reviewing an assessment request. Your task is to determine if you need additional information to provide the most accurate care plan.

ASSESSMENT CONTEXT:
- Target Audience: ${audience}
- Available Context Data: ${JSON.stringify(contextData, null, 2)}

IMAGE ANALYSIS INSTRUCTIONS:
Examine the wound image and current context data. Determine if you need additional clarifying questions to provide an optimal care plan.

Consider asking questions about:
1. UNCLEAR VISUAL ASPECTS: If wound characteristics are ambiguous in the image
2. MISSING CRITICAL CONTEXT: Essential medical history, medications, or care details not provided
3. SAFETY CONCERNS: Potential complications that need immediate clarification
4. TREATMENT HISTORY: Previous treatments, responses, or care attempts
5. SYMPTOM DETAILS: Pain levels, changes, or associated symptoms needing clarification

RESPONSE FORMAT:
Return a JSON object with:
{
  "needsQuestions": boolean,
  "questions": ["question1", "question2", ...],
  "questionTypes": ["clarification", "medical_history", "symptom_detail", ...],
  "reasoning": "Brief explanation of why these questions are needed"
}

GUIDELINES:
- Only ask questions that are truly essential for providing accurate care
- Limit to maximum 3 questions to avoid overwhelming the user
- Ask specific, actionable questions that will directly impact care recommendations
- Consider the target audience when framing questions (family, patient, medical professional)
- If sufficient information exists, return needsQuestions: false

Analyze the image and context, then respond with your assessment.
`;

  try {
    let response: string;
    
    if (model.startsWith('gemini')) {
      response = await callGemini(model, analysisPrompt, imageBase64);
    } else {
      const messages = [
        {
          role: "system",
          content: analysisPrompt
        },
        {
          role: "user",
          content: [
            {
              type: "text",
              text: "Please analyze this wound assessment and determine if additional questions are needed."
            },
            {
              type: "image_url",
              image_url: {
                url: `data:image/jpeg;base64,${imageBase64}`
              }
            }
          ],
        },
      ];
      response = await callOpenAI(model, messages, { type: "json_object" });
    }

    const analysis: QuestionAnalysisResult = JSON.parse(response);

    // If questions are needed, store them in the database
    if (analysis.needsQuestions && analysis.questions.length > 0) {
      for (let i = 0; i < analysis.questions.length; i++) {
        const questionData: InsertAgentQuestion = {
          sessionId,
          userId,
          question: analysis.questions[i],
          questionType: analysis.questionTypes[i] || 'clarification',
          context: JSON.stringify({
            audience,
            model,
            reasoning: analysis.reasoning,
            imageProvided: true
          })
        };
        
        await storage.createAgentQuestion(questionData);
      }
    }

    return analysis;
    
  } catch (error) {
    console.error('Error analyzing assessment for questions:', error);
    // Return safe fallback - no questions needed
    return {
      needsQuestions: false,
      questions: [],
      questionTypes: [],
      reasoning: "Unable to analyze - proceeding with available information"
    };
  }
}

export async function getSessionQuestions(sessionId: string) {
  return await storage.getQuestionsBySession(sessionId);
}

export async function answerSessionQuestion(questionId: number, answer: string) {
  return await storage.answerQuestion(questionId, answer);
}

export async function checkSessionComplete(sessionId: string): Promise<boolean> {
  const unanswered = await storage.getUnansweredQuestions(sessionId);
  return unanswered.length === 0;
}