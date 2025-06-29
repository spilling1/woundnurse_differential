import { useState } from "react";
import { useLocation } from "wouter";
import { Stethoscope, Heart, Shield, Activity, Users } from "lucide-react";
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
    // Navigate to care plan page
    setLocation(`/care-plan?caseId=${data.caseId}`);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-medical-blue/5 to-success-green/5"></div>
        <div className="relative max-w-7xl mx-auto px-4 py-12">
          <div className="text-center mb-12">
            <div className="flex items-center justify-center mb-6">
              <div className="relative">
                <div className="absolute -inset-4 bg-gradient-to-r from-medical-blue to-success-green rounded-full opacity-20 blur-lg"></div>
                <div className="relative bg-white rounded-full p-4 shadow-lg">
                  <Stethoscope className="text-medical-blue text-5xl" />
                </div>
              </div>
              <div className="ml-6">
                <h1 className="text-4xl font-bold text-gray-900 mb-2">Wound Nurses</h1>
                <p className="text-xl text-gray-600">AI-Powered Wound Care Assessment</p>
              </div>
            </div>
            
            <p className="text-lg text-gray-600 max-w-3xl mx-auto mb-8">
              Professional wound assessment with personalized treatment planning. Upload photos, provide context, and receive evidence-based care recommendations tailored to your needs.
            </p>

            {/* Feature Highlights */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 max-w-4xl mx-auto">
              <Card className="border-0 shadow-md hover:shadow-lg transition-shadow">
                <CardContent className="p-6 text-center">
                  <div className="w-12 h-12 bg-medical-blue/10 rounded-full flex items-center justify-center mx-auto mb-3">
                    <Shield className="text-medical-blue text-xl" />
                  </div>
                  <h3 className="font-semibold text-gray-900 mb-1">Secure & Private</h3>
                  <p className="text-sm text-gray-600">HIPAA-compliant processing</p>
                </CardContent>
              </Card>
              
              <Card className="border-0 shadow-md hover:shadow-lg transition-shadow">
                <CardContent className="p-6 text-center">
                  <div className="w-12 h-12 bg-success-green/10 rounded-full flex items-center justify-center mx-auto mb-3">
                    <Activity className="text-success-green text-xl" />
                  </div>
                  <h3 className="font-semibold text-gray-900 mb-1">AI Analysis</h3>
                  <p className="text-sm text-gray-600">Advanced image recognition</p>
                </CardContent>
              </Card>
              
              <Card className="border-0 shadow-md hover:shadow-lg transition-shadow">
                <CardContent className="p-6 text-center">
                  <div className="w-12 h-12 bg-accent-orange/10 rounded-full flex items-center justify-center mx-auto mb-3">
                    <Users className="text-accent-orange text-xl" />
                  </div>
                  <h3 className="font-semibold text-gray-900 mb-1">Multi-Audience</h3>
                  <p className="text-sm text-gray-600">Tailored for everyone</p>
                </CardContent>
              </Card>
              
              <Card className="border-0 shadow-md hover:shadow-lg transition-shadow">
                <CardContent className="p-6 text-center">
                  <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-3">
                    <Heart className="text-purple-600 text-xl" />
                  </div>
                  <h3 className="font-semibold text-gray-900 mb-1">Evidence-Based</h3>
                  <p className="text-sm text-gray-600">Clinical guidelines</p>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>

      {/* Main Assessment Section */}
      <div className="max-w-7xl mx-auto px-4 pb-12">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Image Upload */}
          <div className="lg:col-span-1">
            <Card className="border-0 shadow-lg">
              <CardContent className="p-0">
                <div className="bg-gradient-to-br from-medical-blue to-medical-blue/80 text-white p-6 rounded-t-lg">
                  <div className="flex items-center mb-2">
                    <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center mr-3">
                      <span className="text-sm font-bold">1</span>
                    </div>
                    <h2 className="text-lg font-semibold">Upload Image</h2>
                  </div>
                  <p className="text-blue-100 text-sm">Take or upload a clear photo of the wound</p>
                </div>
                <div className="p-6">
                  <ImageUploadSection 
                    selectedFile={selectedFile}
                    onFileSelect={setSelectedFile}
                  />
                </div>
              </CardContent>
            </Card>
            
            <Card className="border-0 shadow-lg mt-6">
              <CardContent className="p-0">
                <div className="bg-gradient-to-br from-success-green to-success-green/80 text-white p-6 rounded-t-lg">
                  <div className="flex items-center mb-2">
                    <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center mr-3">
                      <span className="text-sm font-bold">2</span>
                    </div>
                    <h2 className="text-lg font-semibold">Configure Assessment</h2>
                  </div>
                  <p className="text-green-100 text-sm">Choose your audience and AI model</p>
                </div>
                <div className="p-6">
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
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Questionnaire */}
          <div className="lg:col-span-2">
            <Card className="border-0 shadow-lg">
              <CardContent className="p-0">
                <div className="bg-gradient-to-br from-accent-orange to-accent-orange/80 text-white p-6 rounded-t-lg">
                  <div className="flex items-center mb-2">
                    <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center mr-3">
                      <span className="text-sm font-bold">3</span>
                    </div>
                    <h2 className="text-lg font-semibold">Provide Context</h2>
                  </div>
                  <p className="text-orange-100 text-sm">Answer questions to improve assessment accuracy</p>
                </div>
                <div className="p-6">
                  <WoundQuestionnaire 
                    onDataChange={setContextData}
                    initialData={contextData}
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}