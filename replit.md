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
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```