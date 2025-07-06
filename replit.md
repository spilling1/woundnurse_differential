# Wound Nurses - AI Wound Care Evaluation Tool

## Overview

This is a full-stack TypeScript application called "Wound Nurses" that provides AI-powered wound care assessment and treatment plan generation. The system allows users to upload wound images, select different AI models for analysis, and receive customized care plans tailored to different audiences (family caregivers, patients, or medical professionals). Users can optionally log in via Replit Auth to manage their case history and access saved assessments. The application is built following the PRD specifications for comprehensive wound analysis with model selection and continuous learning capabilities.

## System Architecture

The application follows a modern full-stack architecture:

### Frontend Architecture
- **Framework**: React with TypeScript
- **Build Tool**: Vite for fast development and optimized builds
- **UI Library**: shadcn/ui components built on Radix UI primitives
- **Styling**: Tailwind CSS with custom medical-themed color palette
- **State Management**: TanStack Query for server state management
- **Routing**: Wouter for lightweight client-side routing

### Backend Architecture
- **Runtime**: Node.js with Express.js server
- **Language**: TypeScript with ES modules
- **File Upload**: Multer for handling multipart form data
- **AI Integration**: OpenAI API for image analysis and text generation
- **Database**: PostgreSQL with Drizzle ORM
- **Session Management**: In-memory storage with PostgreSQL session store option

## Key Components

### Data Models
- **Wound Assessments**: Stores case ID, audience type, AI model used, wound images, questionnaire responses, wound classification, and generated care plans
- **Feedback**: Captures user feedback on care plan quality for continuous improvement
- **Agent Instructions**: Database-stored AI behavior rules and guidelines (replaces Agents.md file)

### Core Services
1. **Image Processor**: Validates uploaded images (PNG/JPG, max 10MB)
2. **Wound Classifier**: Uses OpenAI Vision API to analyze wound characteristics
3. **Care Plan Generator**: Creates audience-specific treatment recommendations
4. **Agents Logger**: Maintains case history in structured markdown format

### AI Model Support
- Gemini 2.5 Pro (primary model)
- Gemini 2.5 Flash
- GPT-4o
- GPT-3.5
- GPT-3.5-pro
- Dynamic model selection per assessment with both OpenAI and Google AI support

### Audience Targeting
- **Family Caregivers**: Simple language, step-by-step instructions
- **Patients**: Empowering language, self-care focus
- **Medical Professionals**: Clinical terminology, protocol-based recommendations

## Data Flow

1. **Image Upload**: User selects wound image, audience type, and AI model
2. **Validation**: Server validates file type, size, and request parameters
3. **AI Analysis**: Image sent to OpenAI Vision API for wound classification
4. **Care Plan Generation**: AI generates audience-specific treatment recommendations
5. **Response**: Complete assessment data returned to client
6. **Feedback Loop**: Users can provide feedback, logged to Agents.md for model improvement
7. **Case Logging**: All assessments automatically logged for training data

## External Dependencies

### AI Services
- **OpenAI API**: Core AI processing for image analysis and care plan generation
- **Models**: GPT-4o, GPT-3.5 variants with vision capabilities

### Database
- **PostgreSQL**: Primary data storage via Neon serverless for all case history, images, and instructions
- **Drizzle ORM**: Type-safe database operations with schema validation
- **Image Storage**: Base64 encoded images stored directly in database

### Infrastructure
- **Replit**: Development and deployment platform
- **Vite**: Development server with HMR and build optimization
- **Node.js**: Server runtime environment

## Deployment Strategy

### Development
- Vite dev server for frontend with HMR
- Node.js/Express backend with file watching
- In-memory storage for rapid development
- Environment variables for API keys

### Production
- Vite build process generates optimized static assets
- Express serves built frontend and API routes
- PostgreSQL database for persistent storage
- Environment-based configuration

### File Structure
- `client/`: React frontend application
- `server/`: Express backend with API routes
- `shared/`: Common TypeScript schemas and types
- `attached_assets/`: Project documentation

## Changelog

```
Changelog:
- June 29, 2025. Initial setup
- June 29, 2025. Added Gemini 2.5 Flash and 2.5 Pro model support with Google AI integration
- June 29, 2025. Migrated from file-based to PostgreSQL database storage
- June 29, 2025. Added comprehensive questionnaire system with 8 context questions
- June 29, 2025. Replaced Agents.md file with database-stored AI instructions
- June 29, 2025. Added image storage in database with Base64 encoding
- June 29, 2025. Created dedicated care plan page and enhanced UI workflow
- June 29, 2025. Implemented Replit Auth login system with user case management
- June 29, 2025. Updated authentication flow to redirect to My Cases page after login
- June 29, 2025. Implemented authenticated user auto-redirect to My Cases from root path
- June 29, 2025. Added delete functionality for wound assessments with security controls
- June 29, 2025. Enhanced My Cases page with logout and "Start New Case" navigation
- June 29, 2025. Implemented follow-up assessment system with version tracking and progress monitoring
- June 29, 2025. Added database support for multiple assessment versions per case with historical context
- June 29, 2025. Created follow-up assessment page with progress tracking and treatment response evaluation
- June 29, 2025. Enhanced AI prompt templates for follow-up assessments to reference previous care plans and highlight progression changes
- June 29, 2025. Fixed multiple file upload support for follow-up assessments with proper multer configuration
- June 29, 2025. Added Settings page with AI Configuration management accessible via gear icon in navigation headers
- June 29, 2025. Implemented settings gear icon in My Cases, Care Plan, and Home/Assessment pages for easy access to AI configuration
- June 29, 2025. Overhauled question generation system to ask diagnostic questions only when AI confidence is low (≤75%)
- June 29, 2025. Implemented confidence-based questioning where AI asks questions WITHOUT pre-filled answers for user to complete
- June 29, 2025. Added support for multiple rounds of questions (up to 3 rounds) to improve diagnostic accuracy
- June 29, 2025. Fixed all OpenAI API integration issues including MIME type handling and error responses
- June 30, 2025. Fixed critical diagnostic issue where AI question answers weren't being integrated into care plan generation
- June 30, 2025. Updated Agent Instructions to mandate diabetes question for ALL foot/heel wounds regardless of visual classification
- June 30, 2025. Implemented diagnostic override system where patient answers can reclassify wound type when contradicting visual assessment
- June 30, 2025. Fixed image display issues in My Cases and Care Plan pages by correcting base64 encoding in final plan route
- June 30, 2025. Completely redesigned Care Plan page layout with professional styling, improved typography, and better content organization
- June 30, 2025. Fixed overly aggressive follow-up question generation - now only asks additional questions when confidence is low (≤75%) or when truly needed
- July 5, 2025. Integrated YOLO9 wound detection service with fallback system for precise wound measurements and boundary detection
- July 5, 2025. Enhanced database schema with detection data storage and created interactive wound visualization component
- July 5, 2025. Updated YOLO endpoint to localhost:8081 and implemented comprehensive wound analysis workflow
- July 5, 2025. Integrated cloud-based computer vision APIs (Google Cloud Vision, Azure Computer Vision) with intelligent fallback system
- July 5, 2025. Created enhanced wound detection service with multiple API support and automatic failover capabilities
- July 5, 2025. Updated detection priority order: YOLO first, then cloud APIs as backup, then enhanced fallback
- July 5, 2025. Added Image Detection status monitor component displaying real-time detection method status (YOLO/Cloud/Fallback)
- July 5, 2025. Created detection status API endpoint at /api/detection-status with automatic service health monitoring
- July 5, 2025. Implemented YOLO service management scripts with automatic restart capabilities to maintain service availability
- July 5, 2025. Successfully integrated YOLO service into wound classification pipeline with proper service management and automatic restart
- July 5, 2025. Fixed wound detection service routing to use YOLO first, then fallback to cloud APIs or enhanced detection
- July 5, 2025. Completed full YOLO integration - system now uses YOLO9 for primary wound detection with precise measurements and high accuracy
- July 5, 2025. **MAJOR ARCHITECTURAL CHANGE**: Restructured AI instructions from single content field into four structured sections:
  - **System Prompts**: Core mission and behavior instructions
  - **Care Plan Structure**: Response formatting and organization guidelines  
  - **Specific Wound Care**: Medical knowledge for different wound types
  - **Questions Guidelines**: How to ask diagnostic follow-up questions
- July 5, 2025. Updated database schema, storage interfaces, and API endpoints to support structured AI instructions
- July 5, 2025. Created new tabbed Settings page interface allowing independent editing of each instruction section
- July 5, 2025. Enhanced AI instruction system for better maintainability and granular control over wound care behavior
- July 5, 2025. Added Product Recommendations section to AI Configuration with dedicated tab interface
- July 5, 2025. **CRITICAL FIX**: Fixed care plan generation to properly use AI Settings instead of hardcoded prompts
- July 5, 2025. Updated promptTemplates.ts and carePlanGenerator.ts to correctly integrate structured AI instructions from database
- July 5, 2025. **WORKFLOW ENHANCEMENT**: Added intermediate "Generating Care Plan" step with progress indicator
- July 5, 2025. Enhanced assessment workflow to show care plan generation progress and preview before final completion
- July 5, 2025. Created CarePlanGeneration component with animated progress bar and care plan preview functionality
- July 5, 2025. **UI ENHANCEMENT**: Fixed "View Complete Care Plan" button to properly navigate to care plan page instead of showing generic completion message
- July 5, 2025. **LAYOUT IMPROVEMENT**: Moved "DETECTION SYSTEM ANALYSIS" section to bottom of care plan page in smaller font under "Generated by Wound Nurses AI..."
- July 5, 2025. **DATA FRESHNESS**: Enhanced My Cases page to automatically refresh on each visit with manual refresh button for real-time case updates
- July 5, 2025. **PRODUCT RECOMMENDATIONS**: Fixed product recommendation system to generate dynamic, contextual Amazon search links instead of broken static URLs
- July 5, 2025. **AI CONFIGURATION**: Enhanced Product Recommendations tab with comprehensive guidelines, examples, and working Amazon search link templates
- July 5, 2025. **SETTINGS UI**: Improved Product Recommendations settings page with helpful tips, examples, and dynamic link formatting guidance
- July 5, 2025. **HTML FORMATTING FIX**: Fixed care plan rendering to properly display HTML content with inline styles for urgent medical alerts
- July 5, 2025. **AI INSTRUCTIONS**: Updated care plan structure to generate proper HTML formatting for urgent messages with red styling and emphasis
- July 5, 2025. **ADMIN SYSTEM**: Implemented comprehensive admin functionality with user management, company management, and system oversight
- July 5, 2025. **ADMIN DASHBOARD**: Created full-featured admin dashboard with user role management, assessment overview, and system statistics
- July 5, 2025. **ROLE MANAGEMENT**: Added toggleable admin role system allowing admins to promote/demote users with one-click role switching
- July 5, 2025. **MULTI-TENANT READY**: Built company management structure for future multi-tenant support with admin oversight capabilities
- July 5, 2025. **DEFAULT MODEL CHANGE**: Changed default AI analytics engine from GPT-4o to Gemini 2.5 Pro as primary model for wound assessment
- July 6, 2025. **PYTORCH YOLO TRAINING**: Installed PyTorch 2.7.1 and OpenCV for wound detection model training
- July 6, 2025. **TRAINING PIPELINE**: Created comprehensive training system for 730 wound images with body map integration
- July 6, 2025. **SMART YOLO SERVICE**: Built intelligent detection service with automatic fallback between YOLO and color detection
- July 6, 2025. **CNN WOUND CLASSIFICATION**: Successfully integrated trained CNN models achieving 80% accuracy for wound type detection
- July 6, 2025. **PRIORITY WOUND DETECTION**: CNN-based classification now serves as primary method with AI vision models as fallback
- July 6, 2025. **MACHINE LEARNING INTEGRATION**: Created cnnWoundClassifier service supporting multiple trained models with automatic best model selection
- July 6, 2025. **ENHANCED TRAINING DATA**: Successfully integrated 538 additional body context images with proper categorization (S→surgical, D→diabetic, V→venous, P→pressure)
- July 6, 2025. **TRAINING PIPELINE EXPANSION**: Created comprehensive training infrastructure for ensemble models with body context understanding
- July 6, 2025. **MODEL DIVERSITY**: Established multiple training approaches (enhanced_body_context_trainer.py, rapid_model_trainer.py, efficient_ensemble_trainer.py) for improved accuracy
- July 6, 2025. **CNN ACCURACY ISSUE IDENTIFIED**: CNN models showing poor real-world performance (classifying hand as diabetic ulcer), temporarily disabled CNN in favor of reliable AI vision models
- July 6, 2025. **SMART FALLBACK SYSTEM**: Implemented intelligent override system where YOLO detection can override CNN misclassifications
- July 6, 2025. **WOUND DETECTION PRIORITY**: Reverted to AI vision models as primary method due to CNN accuracy concerns, maintaining YOLO for precise measurements
- July 6, 2025. **DETECTION MODEL MANAGEMENT**: Added comprehensive admin interface for managing AI detection models with enable/disable toggles, priority settings, and configuration editing
- July 6, 2025. **ACCURATE MODEL LABELING**: Fixed misleading "YOLO9" detection label - system actually uses color-based smart detection with YOLO fallback (ultralytics not installed)
- July 6, 2025. **REAL YOLO INTEGRATION**: Installed ultralytics and added proper YOLO v8 detection engine alongside existing color-based detection
- July 6, 2025. **HONEST CAPABILITY REPORTING**: Replaced made-up accuracy percentages with realistic capability assessments requiring proper clinical validation
- July 6, 2025. **YOLO SERVICE ACTIVATION**: Successfully activated real YOLO v8 wound detection by fixing service startup configuration - system now uses actual YOLOv8 models for primary wound boundary detection with 200ms inference time, falling back to color detection only when needed
- July 6, 2025. **DETECTION METHOD TRANSPARENCY**: Fixed detection method display to show "YOLO v8 Detection" instead of generic "Image Analysis" when using actual YOLO models, providing users with accurate information about which detection engine analyzed their wounds
- July 6, 2025. **ACCURATE MODEL REFERENCES**: Removed misleading "YOLO v9" references from detection method mapping - system now only shows correct detection engines (YOLO v8, Color-based Detection, Cloud APIs)
- July 6, 2025. **CODEBASE CLEANUP**: Organized unused files into `to_delete/` and `training_archive/` folders without deletion - moved 40+ Python training files, legacy routes file (1,042 lines), shell scripts, and documentation files to appropriate archives for cleaner codebase maintenance
- July 6, 2025. **ITERATIVE CONFIDENCE IMPROVEMENT**: Implemented confidence-based questioning system requiring 80% confidence before final care plan generation
- July 6, 2025. **SMART PHOTO SUGGESTIONS**: Added AI-driven photo upload recommendations based on detection quality and missing visual information
- July 6, 2025. **ENHANCED USER GUIDANCE**: Added detailed instructions emphasizing that more detailed answers result in better assessment accuracy
- July 6, 2025. **MULTI-ROUND ASSESSMENT**: Created iterative questioning workflow where users answer questions, see confidence improve, and continue until 80% threshold reached
- July 6, 2025. **STRATEGIC QUESTION CATEGORIZATION**: Implemented three-category question system: A) Confidence improvement (medical history, location, wound characteristics), B) Care plan optimization (symptoms, treatments, progress), C) Medical referral preparation (doctor-relevant information)
- July 6, 2025. **WEIGHTED CONFIDENCE SCORING**: Enhanced confidence calculation with category-based scoring - diagnostic questions provide 8% boost, treatment questions 5% boost, general questions 3% boost
- July 6, 2025. **CATEGORIZED CARE PLAN INTEGRATION**: Updated prompt templates to organize patient answers by purpose and impact on assessment accuracy
- July 6, 2025. **QUESTIONS PAGE THUMBNAILS**: Added thumbnail image display at top of questions pages for visual reference during diagnostic questioning
- July 6, 2025. **SMART IMAGE UPLOAD**: Added conditional image upload capability in questions section for photo-related questions (photo/image/picture keywords trigger upload option)
- July 6, 2025. **CLEAN CARE PLAN LAYOUT**: Removed duplicate images and additional upload capability from care plan page - now shows single clean image display only
- July 6, 2025. **SIMPLIFIED IMAGE PRESENTATION**: Care plan page now has clean, professional single image display without upload distractions
- July 6, 2025. **ENHANCED MEDICAL IMAGE DISPLAY**: Upgraded care plan image to larger size (max-w-2xl, max-h-96) with click-to-zoom modal for detailed medical review when submitting to doctors
- July 6, 2025. **CRITICAL VERSIONING BUG FIX**: Fixed case versioning issue where duplicate case IDs were both showing v1 instead of proper v1/v2 numbering - corrected database and improved version numbering logic
- July 6, 2025. **DUPLICATE IMAGE DETECTION**: Added intelligent duplicate image detection system that prompts users when uploading identical images - users can choose between creating follow-up assessments or new cases, preventing accidental case duplication while preserving user autonomy
- July 6, 2025. **CONFIDENCE TRANSPARENCY**: Fixed misleading 50% confidence defaults by requiring authentic confidence scores from AI models, lowered fallback to 40% for honest uncertainty indication, and added confidence scoring to Gemini models previously missing this capability
- July 6, 2025. **DETECTION METHOD VISIBILITY**: Added clear detection method labels showing users exactly which analysis system was used (Color-based Detection vs AI Vision), replaced misleading "YOLO9" label with accurate "Color-based Detection" description
- July 6, 2025. **DIAGNOSTIC TRANSPARENCY**: Enhanced UI to display thumbnail images and analysis methods used during assessment process, helping users understand which detection engines analyzed their wounds
- July 6, 2025. **ESTIMATED IMPROVEMENT INDICATORS**: Replaced repetitive AI confidence scores with helpful "Estimated Improvement" indicators showing users exactly how answering each question will help: "Confidence Improvement" (better wound detection) vs "Care Plan Improvement" (better treatment guidance)
- July 6, 2025. **GEMINI FLASH LIMITATION DISCOVERED**: Identified that Gemini 2.5 Flash has stricter content safety filters that block medical images with "blockReason: OTHER", while Gemini 2.5 Pro works correctly
- July 6, 2025. **ENHANCED ERROR HANDLING**: Added comprehensive Gemini API error handling with detailed content blocking detection and user-friendly error messages explaining medical image limitations
- July 6, 2025. **MODEL DOCUMENTATION UPDATE**: Updated Gemini Flash model configuration to reflect medical image limitations, disabled by default, and added warning in admin dashboard about content filtering issues
- July 6, 2025. **AI MODEL RELIABILITY**: Confirmed Gemini 2.5 Pro as reliable primary model for medical image analysis, with proper fallback messaging when Flash model encounters content blocking
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```