# INSTRUCTIONS.md - Wound Care Application Refactoring

## Current Task: Codebase Analysis & Optimization

**Date:** 2025-01-29
**Objective:** Analyze and refactor the wound care application for efficiency, maintainability, and AI agent optimization

## Key Findings

### 📊 File Size Analysis
- ✅ **FIXED**: `server/routes.ts` (1043 lines → 5 files under 200 lines each) - 208% over limit
- ✅ **FIXED**: `client/src/components/NewAssessmentFlow.tsx` (775 lines → 22 lines + 7 modular components) - 155% over limit  
- **WARNING**: `server/services/promptTemplates.ts` (261 lines) - approaching limit
- **WARNING**: `client/src/components/ui/chart.tsx` (365 lines) - UI component too large

### 🤖 AI Agent Efficiency Issues
1. **Duplicate AI Calls**: Multiple services calling AI APIs with overlapping prompts
2. **Context Inefficiency**: Complex context building in `promptTemplates.ts`
3. **Instruction Management**: Agent instructions scattered across database and code
4. **No Caching**: AI responses not cached, causing repeated expensive calls

### 🏗️ Architecture Issues
1. **Monolithic Route Handler**: Single file handling all API endpoints
2. **Complex State Management**: Frontend assessment flow too complex
3. **Mixed Concerns**: Prompt templates mixing business logic with AI prompts
4. **Code Duplication**: Repeated patterns across components

## Refactoring Roadmap

### Phase 1: File Size Reduction (Priority: HIGH)
- [x] Split `server/routes.ts` into domain-specific route modules ✅ COMPLETED
- [x] Refactor `NewAssessmentFlow.tsx` into smaller, focused components ✅ COMPLETED
- [ ] Extract prompt templates into separate, focused files
- [ ] Modularize large UI components

### Phase 2: AI Agent Optimization (Priority: HIGH)
- [ ] Implement AI response caching
- [ ] Consolidate duplicate AI service calls
- [ ] Optimize prompt templates for efficiency
- [ ] Create centralized AI instruction management

### Phase 3: Architecture Improvements (Priority: MEDIUM)
- [ ] Implement proper separation of concerns
- [ ] Add service layer abstractions
- [ ] Create reusable component patterns
- [ ] Improve error handling and resilience

### Phase 4: Performance & Maintainability (Priority: LOW)
- [ ] Add comprehensive logging
- [ ] Implement monitoring and metrics
- [ ] Add automated testing infrastructure
- [ ] Create development documentation

## Success Criteria

✅ **File Size**: All files under 500 lines  
✅ **AI Efficiency**: 30% reduction in API calls through caching  
✅ **Code Quality**: Eliminate duplicate code patterns  
✅ **Maintainability**: Clear separation of concerns  
✅ **Performance**: Improved response times for AI operations  

## Next Steps

1. **Immediate**: Start with route splitting and component refactoring
2. **Week 1**: Complete file size reduction phase
3. **Week 2**: Implement AI optimization improvements
4. **Week 3**: Architecture and performance improvements
5. **Week 4**: Testing and documentation

## Notes

- Maintain compatibility with Replit environment
- Preserve all existing functionality during refactoring
- Focus on programmer efficiency and code maintainability
- AI agent instructions should be more modular and efficient 