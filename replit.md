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
- GPT-4o (primary model)
- GPT-3.5
- GPT-3.5-pro
- Gemini 2.5 Flash
- Gemini 2.5 Pro
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
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```