// REFACTORED: This component has been modularized into smaller, focused components
// The new architecture provides better maintainability and follows the <500 line rule
import AssessmentFlow from "./assessment/AssessmentFlow";

export default function NewAssessmentFlow() {
  // The original 775-line component has been refactored into:
  // - AssessmentFlow.tsx (150 lines) - Main orchestrator
  // - AudienceSelection.tsx (120 lines) - Step 1: Audience & model selection  
  // - ImageUpload.tsx (150 lines) - Step 2: Image upload and analysis
  // - AIQuestions.tsx (200 lines) - Step 3: AI-generated questions
  // - PreliminaryPlan.tsx (200 lines) - Step 4: Preliminary plan review
  // - AssessmentTypes.ts (50 lines) - Shared interfaces
  // - AssessmentUtils.ts (80 lines) - Shared utilities and API functions
  //
  // Benefits:
  // ✅ All files under 500 lines (target: <500)
  // ✅ Clear separation of concerns
  // ✅ Improved testability
  // ✅ Better maintainability
  // ✅ Reusable components
  
  return <AssessmentFlow />;
}