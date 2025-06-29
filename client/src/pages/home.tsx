import { useState } from "react";
import { useLocation } from "wouter";
import { Stethoscope, Upload, Settings, ClipboardList } from "lucide-react";
import ImageUploadSection from "@/components/ImageUploadSection";
import ConfigurationPanel from "@/components/ConfigurationPanel";
import WoundQuestionnaire, { WoundContextData } from "@/components/WoundQuestionnaire";
import { Card, CardContent } from "@/components/ui/card";

export default function Home() {
  const [, setLocation] = useLocation();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
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

  const handleStartAssessment = () => setIsProcessing(true);
  
  const handleAssessmentComplete = (data: any) => {
    setAssessmentData(data);
    setIsProcessing(false);
    setLocation(`/care-plan?caseId=${data.caseId}`);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="text-center">
            <div className="flex items-center justify-center mb-4">
              <div className="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center mr-4">
                <Stethoscope className="text-white text-2xl" />
              </div>
              <h1 className="text-3xl font-bold text-gray-900">Wound Nurses</h1>
            </div>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Wound care assessment and treatment planning
            </p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Step 1 - Upload Image */}
          <Card className="shadow-lg">
            <div className="bg-blue-600 text-white p-6 rounded-t-lg">
              <div className="flex items-center">
                <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center mr-4">
                  <span className="text-lg font-bold">1</span>
                </div>
                <div className="flex-1">
                  <h2 className="text-xl font-semibold">Upload Image</h2>
                  <p className="text-blue-100 text-sm">Take or upload a clear photo</p>
                </div>
                <Upload className="text-2xl opacity-75" />
              </div>
            </div>
            <CardContent className="p-6">
              <ImageUploadSection 
                selectedFile={selectedFile}
                onFileSelect={setSelectedFile}
              />
            </CardContent>
          </Card>

          {/* Step 2 - Configure */}
          <Card className="shadow-lg">
            <div className="bg-green-600 text-white p-6 rounded-t-lg">
              <div className="flex items-center">
                <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center mr-4">
                  <span className="text-lg font-bold">2</span>
                </div>
                <div className="flex-1">
                  <h2 className="text-xl font-semibold">Configure Assessment</h2>
                  <p className="text-green-100 text-sm">Choose your preferences</p>
                </div>
                <Settings className="text-2xl opacity-75" />
              </div>
            </div>
            <CardContent className="p-6">
              <ConfigurationPanel
                audience={audience}
                model={model}
                onAudienceChange={setAudience}
                onModelChange={setModel}
                selectedFile={selectedFile}
                isProcessing={isProcessing}
                contextData={contextData}
                onStartAssessment={handleStartAssessment}
                onAssessmentComplete={handleAssessmentComplete}
              />
            </CardContent>
          </Card>

          {/* Step 3 - Questionnaire */}
          <Card className="shadow-lg">
            <div className="bg-purple-600 text-white p-6 rounded-t-lg">
              <div className="flex items-center">
                <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center mr-4">
                  <span className="text-lg font-bold">3</span>
                </div>
                <div className="flex-1">
                  <h2 className="text-xl font-semibold">Provide Context</h2>
                  <p className="text-purple-100 text-sm">Answer assessment questions</p>
                </div>
                <ClipboardList className="text-2xl opacity-75" />
              </div>
            </div>
            <CardContent className="p-6">
              <WoundQuestionnaire 
                onDataChange={setContextData}
                initialData={contextData}
              />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}