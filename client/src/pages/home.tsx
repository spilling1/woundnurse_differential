import { useState } from "react";
import { useLocation } from "wouter";
import { Stethoscope, Circle, HelpCircle, Plus, LogOut } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/button";
import ImageUploadSection from "@/components/ImageUploadSection";
import ConfigurationPanel from "@/components/ConfigurationPanel";
import WoundQuestionnaire, { WoundContextData } from "@/components/WoundQuestionnaire";

export default function Home() {
  const [, setLocation] = useLocation();
  const { isAuthenticated } = useAuth();
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [audience, setAudience] = useState<'family' | 'patient' | 'medical'>('family');
  const [model, setModel] = useState<'gpt-4o' | 'gpt-3.5' | 'gpt-3.5-pro' | 'gemini-2.5-flash' | 'gemini-2.5-pro'>('gpt-4o');
  const [assessmentData, setAssessmentData] = useState<any>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [contextData, setContextData] = useState<WoundContextData>({
    woundOrigin: '',
    medicalHistory: '',
    woundChanges: '',
    currentCare: '',
    woundPain: '',
    supportAtHome: '',
    mobilityStatus: '',
    nutritionStatus: '',
    stressLevel: '',
    comorbidities: '',
    age: '',
    obesity: '',
    medications: '',
    alcoholUse: '',
    smokingStatus: '',
    frictionShearing: '',
    knowledgeDeficits: '',
    woundSite: ''
  });

  return (
    <div className="font-inter bg-bg-light min-h-screen">
      {/* Navigation Header */}
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <div className="flex-shrink-0 flex items-center">
                <Stethoscope className="text-medical-blue text-2xl mr-3" />
                <h1 className="text-xl font-semibold text-gray-900">Wound Nurses</h1>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-500">
                <span className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs font-medium">
                  <Circle className="inline w-2 h-2 mr-1 fill-current" />
                  System Online
                </span>
              </div>
              <button 
                onClick={() => window.location.href = '/agents'}
                className="text-gray-600 hover:text-gray-800 text-sm font-medium"
              >
                AI Configuration
              </button>
              
              {/* Authentication-aware navigation */}
              {isAuthenticated ? (
                <div className="flex items-center space-x-2">
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => setLocation('/my-cases')}
                    className="border-medical-blue text-medical-blue hover:bg-medical-blue hover:text-white"
                  >
                    My Cases
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => window.location.href = "/api/logout"}
                  >
                    <LogOut className="mr-2 h-4 w-4" />
                    Log Out
                  </Button>
                </div>
              ) : (
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => setLocation('/start-assessment')}
                  className="border-medical-blue text-medical-blue hover:bg-medical-blue hover:text-white"
                >
                  Sign In
                </Button>
              )}
              
              <button className="text-gray-400 hover:text-gray-500">
                <HelpCircle className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Main Content Container */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Left Column - Upload and Configuration */}
          <div className="lg:col-span-1">
            <ImageUploadSection 
              selectedFiles={selectedFiles}
              onFilesSelect={setSelectedFiles}
            />
            
            <ConfigurationPanel
              audience={audience}
              model={model}
              onAudienceChange={setAudience}
              onModelChange={setModel}
              selectedFile={selectedFiles[0] || null}
              isProcessing={isProcessing}
              contextData={contextData}
              onStartAssessment={() => setIsProcessing(true)}
              onAssessmentComplete={(data) => {
                setAssessmentData(data);
                setIsProcessing(false);
                // Navigate to care plan page
                setLocation(`/care-plan?caseId=${data.caseId}`);
              }}
            />
          </div>

          {/* Right Column - Questionnaire */}
          <div className="lg:col-span-2">
            <WoundQuestionnaire 
              onDataChange={setContextData}
              initialData={contextData}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
