import { useState } from "react";
import { Stethoscope, Circle, HelpCircle } from "lucide-react";
import ImageUploadSection from "@/components/ImageUploadSection";
import ConfigurationPanel from "@/components/ConfigurationPanel";
import AssessmentResults from "@/components/AssessmentResults";
import CarePlanSection from "@/components/CarePlanSection";
import SystemStatus from "@/components/SystemStatus";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [audience, setAudience] = useState<'family' | 'patient' | 'medical'>('family');
  const [model, setModel] = useState<'gpt-4o' | 'gpt-3.5' | 'gpt-3.5-pro'>('gpt-4o');
  const [assessmentData, setAssessmentData] = useState<any>(null);
  const [isProcessing, setIsProcessing] = useState(false);

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
              selectedFile={selectedFile}
              onFileSelect={setSelectedFile}
            />
            
            <ConfigurationPanel
              audience={audience}
              model={model}
              onAudienceChange={setAudience}
              onModelChange={setModel}
              selectedFile={selectedFile}
              isProcessing={isProcessing}
              onStartAssessment={() => setIsProcessing(true)}
              onAssessmentComplete={(data) => {
                setAssessmentData(data);
                setIsProcessing(false);
              }}
            />
          </div>

          {/* Right Column - Results and Analysis */}
          <div className="lg:col-span-2">
            <AssessmentResults 
              assessmentData={assessmentData}
              isProcessing={isProcessing}
            />
            
            <CarePlanSection
              assessmentData={assessmentData}
              model={model}
              isProcessing={isProcessing}
            />
          </div>
        </div>

        <SystemStatus />
      </div>
    </div>
  );
}
