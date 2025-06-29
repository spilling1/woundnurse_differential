import fs from 'fs/promises';
import path from 'path';
import type { WoundAssessment, Feedback } from '@shared/schema';

const AGENTS_FILE = path.join(process.cwd(), 'Agents.md');

export async function logToAgents(
  assessment: WoundAssessment, 
  classification: any, 
  carePlan: string, 
  feedback?: Feedback
): Promise<void> {
  try {
    const entry = formatAgentsEntry(assessment, classification, carePlan, feedback);
    
    // Read existing content or create new file
    let existingContent = '';
    try {
      existingContent = await fs.readFile(AGENTS_FILE, 'utf-8');
    } catch (error) {
      // File doesn't exist, create header
      existingContent = '# Wound Care AI - Case Log\n\nThis file tracks all wound assessments and feedback for continuous learning.\n\n';
    }
    
    // Append new entry
    const updatedContent = existingContent + entry + '\n';
    await fs.writeFile(AGENTS_FILE, updatedContent);
    
  } catch (error: any) {
    console.error('Failed to log to Agents.md:', error);
    // Don't throw error as this shouldn't break the main flow
  }
}

function formatAgentsEntry(
  assessment: WoundAssessment, 
  classification: any, 
  carePlan: string, 
  feedback?: Feedback
): string {
  const timestamp = new Date().toISOString();
  
  let entry = `## Case: ${assessment.caseId}
**Timestamp:** ${timestamp}  
**Model:** ${assessment.model}  
**Audience:** ${assessment.audience}  
**Version:** ${assessment.version}  

**Wound Classification:**
- Type: ${classification.woundType}
- Stage: ${classification.stage}
- Size: ${classification.size}
- Wound Bed: ${classification.woundBed}
- Exudate: ${classification.exudate}
- Location: ${classification.location}
- Infection Signs: ${classification.infectionSigns.join(', ') || 'None'}

**Care Plan:** ${carePlan.replace(/\n/g, '\n> ')}`;

  if (feedback) {
    entry += `\n\n**Feedback:** ${feedback.feedbackType}`;
    if (feedback.comments) {
      entry += `\n**Comments:** ${feedback.comments}`;
    }
    entry += `\n**Feedback Date:** ${feedback.createdAt?.toISOString()}`;
  }

  entry += '\n\n---\n';
  
  return entry;
}
