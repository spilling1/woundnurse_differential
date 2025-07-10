import { promises as fs } from 'fs';
import path from 'path';

interface ClassificationLogEntry {
  caseId: string;
  timestamp: string;
  userId: string;
  email: string;
  woundType: string;
  confidence: number;
  aiModel: string;
  reasoning: string;
  detectionMethod: string;
  yoloInfluence?: {
    enabled: boolean;
    detectionFound: boolean;
    yoloConfidence?: number;
    confidenceChange?: string;
  };
  independentClassification?: {
    woundType: string;
    confidence: number;
  };
  finalDecision: string;
}

export class WhyClassificationLogger {
  private logFilePath: string;

  constructor() {
    this.logFilePath = path.join(process.cwd(), 'WhyClassification.md');
  }

  /**
   * Extract reasoning from AI response text
   */
  private extractReasoning(responseText: string): string {
    try {
      // Try to parse as JSON first (structured AI responses)
      let parsedResponse;
      try {
        parsedResponse = JSON.parse(responseText);
      } catch {
        // Not JSON, treat as plain text
        parsedResponse = null;
      }

      let reasoning = '';

      // If structured response, extract reasoning fields
      if (parsedResponse && typeof parsedResponse === 'object') {
        const reasoningFields = [
          'reasoning', 'explanation', 'rationale', 'additionalObservations',
          'imageAnalysis', 'assessment', 'clinicalFindings', 'visualAnalysis'
        ];

        for (const field of reasoningFields) {
          if (parsedResponse[field] && typeof parsedResponse[field] === 'string') {
            reasoning += parsedResponse[field] + ' ';
          }
        }
      }

      // If no structured reasoning found, extract from text patterns
      if (!reasoning.trim()) {
        const reasoningPatterns = [
          /(?:reasoning|explanation|rationale)[:\s]+"([^"]+)"/gi,
          /(?:because|due to|given that|considering that)[:\s]+(.+?)(?:\n|\.|\.|$)/gi,
          /(?:based on|according to|analysis shows)[:\s]+(.+?)(?:\n|\.|\.|$)/gi,
          /(?:this appears to be|i assess this as|classification)[:\s]+(.+?)(?:\n|\.|\.|$)/gi,
          /(?:the image shows|visual characteristics include)[:\s]+(.+?)(?:\n|\.|\.|$)/gi,
          /(?:key findings|observations|clinical signs)[:\s]+(.+?)(?:\n|\.|\.|$)/gi,
          /(?:differential diagnosis|assessment)[:\s]+(.+?)(?:\n|\.|\.|$)/gi
        ];

        for (const pattern of reasoningPatterns) {
          const matches = responseText.match(pattern);
          if (matches && matches.length > 0) {
            reasoning += matches.map(match => match.replace(pattern, '$1')).join(' ') + ' ';
          }
        }
      }

      // Fallback: extract first meaningful sentences
      if (!reasoning.trim()) {
        const sentences = responseText
          .split(/[.!?]+/)
          .filter(s => s.trim().length > 10)
          .slice(0, 3);
        reasoning = sentences.join('. ').trim();
      }

      // Clean up the reasoning
      reasoning = reasoning
        .replace(/^\s*[{}"]+\s*/, '') // Remove leading JSON artifacts
        .replace(/\s*[{}"]+\s*$/, '') // Remove trailing JSON artifacts
        .replace(/reasoning[:\s]+/gi, '')
        .replace(/explanation[:\s]+/gi, '')
        .replace(/rationale[:\s]+/gi, '')
        .replace(/"/g, '')
        .replace(/\s+/g, ' ')
        .trim();

      return reasoning || 'No specific reasoning provided in AI response';
    } catch (error) {
      console.error('Error extracting reasoning:', error);
      return 'Error extracting reasoning from AI response';
    }
  }

  /**
   * Extract detailed reasoning from AI response with more context
   */
  private extractDetailedReasoning(responseText: string): string {
    try {
      // Look for reconsideration reasoning (YOLO integration responses)
      const reconsiderationMatch = responseText.match(/Given my initial assessment.*?reconsider[^.]*\./s);
      if (reconsiderationMatch) {
        return reconsiderationMatch[0].trim();
      }

      // Look for assessment reasoning
      const assessmentMatch = responseText.match(/(?:assessment|analysis|reasoning)[:\s]+(.+?)(?:\n\n|\.|$)/s);
      if (assessmentMatch) {
        return assessmentMatch[1].trim();
      }

      // Fallback to original extraction
      return this.extractReasoning(responseText);
    } catch (error) {
      console.error('Error extracting detailed reasoning:', error);
      return this.extractReasoning(responseText);
    }
  }

  /**
   * Log a classification decision with reasoning
   */
  async logClassification(data: {
    caseId: string;
    userId: string;
    email: string;
    woundType: string;
    confidence: number;
    aiModel: string;
    aiResponse: string;
    detectionMethod: string;
    yoloData?: {
      enabled: boolean;
      detectionFound: boolean;
      yoloConfidence?: number;
      originalConfidence?: number;
    };
    independentClassification?: {
      woundType: string;
      confidence: number;
    };
    extractedReasoning?: string;
  }): Promise<void> {
    try {
      const reasoning = data.extractedReasoning || this.extractDetailedReasoning(data.aiResponse);
      
      let finalDecision = `AI classified as ${data.woundType} with ${Math.round(data.confidence * 100)}% confidence`;
      
      // Add YOLO influence analysis
      let yoloInfluence = undefined;
      if (data.yoloData) {
        yoloInfluence = {
          enabled: data.yoloData.enabled,
          detectionFound: data.yoloData.detectionFound,
          yoloConfidence: data.yoloData.yoloConfidence,
          confidenceChange: data.yoloData.originalConfidence 
            ? (data.confidence > data.yoloData.originalConfidence ? 'Increased' : 
               data.confidence < data.yoloData.originalConfidence ? 'Decreased' : 'Unchanged')
            : undefined
        };
        
        if (data.yoloData.enabled && data.yoloData.detectionFound) {
          finalDecision += `. YOLO detection ${yoloInfluence.confidenceChange?.toLowerCase() || 'influenced'} confidence`;
        } else if (data.yoloData.enabled && !data.yoloData.detectionFound) {
          finalDecision += `. YOLO found no wounds, AI assessment stands`;
        }
      }

      const logEntry: ClassificationLogEntry = {
        caseId: data.caseId,
        timestamp: new Date().toLocaleString(),
        userId: data.userId,
        email: data.email,
        woundType: data.woundType,
        confidence: Math.round(data.confidence * 100),
        aiModel: data.aiModel,
        reasoning,
        detectionMethod: data.detectionMethod,
        yoloInfluence,
        independentClassification: data.independentClassification ? {
          woundType: data.independentClassification.woundType,
          confidence: Math.round(data.independentClassification.confidence * 100)
        } : undefined,
        finalDecision
      };

      await this.appendToLog(logEntry);
      console.log(`WhyClassificationLogger: Logged classification for case ${data.caseId}`);
    } catch (error) {
      console.error('WhyClassificationLogger: Error logging classification:', error);
    }
  }

  /**
   * Append log entry to markdown file
   */
  private async appendToLog(entry: ClassificationLogEntry): Promise<void> {
    try {
      let logContent = `
---
## Case: ${entry.caseId}
**User:** ${entry.email}  
**Date:** ${entry.timestamp}  
**Wound Type:** ${entry.woundType}  
**Confidence:** ${entry.confidence}%  
**AI Model:** ${entry.aiModel}  
**Detection Method:** ${entry.detectionMethod}  

### AI Reasoning:
${entry.reasoning}

### Classification Process:
`;

      if (entry.independentClassification) {
        logContent += `
**Independent Classification:** ${entry.independentClassification.woundType} (${entry.independentClassification.confidence}% confidence)  
`;
      }

      if (entry.yoloInfluence) {
        logContent += `
**YOLO Status:** ${entry.yoloInfluence.enabled ? 'Enabled' : 'Disabled'}  
`;
        if (entry.yoloInfluence.enabled) {
          logContent += `**YOLO Detection:** ${entry.yoloInfluence.detectionFound ? 'Found' : 'Not Found'}  
`;
          if (entry.yoloInfluence.detectionFound && entry.yoloInfluence.yoloConfidence) {
            logContent += `**YOLO Confidence:** ${Math.round(entry.yoloInfluence.yoloConfidence * 100)}%  
`;
          }
          if (entry.yoloInfluence.confidenceChange) {
            logContent += `**Confidence Change:** ${entry.yoloInfluence.confidenceChange}  
`;
          }
        }
      }

      logContent += `
**Final Decision:** ${entry.finalDecision}

---

`;

      await fs.appendFile(this.logFilePath, logContent, 'utf8');
    } catch (error) {
      console.error('Error appending to log file:', error);
    }
  }

  /**
   * Get recent classification logs
   */
  async getRecentLogs(limit: number = 10): Promise<string> {
    try {
      const content = await fs.readFile(this.logFilePath, 'utf8');
      return content;
    } catch (error) {
      console.error('Error reading log file:', error);
      return 'Error reading classification logs';
    }
  }
}

export const whyClassificationLogger = new WhyClassificationLogger();