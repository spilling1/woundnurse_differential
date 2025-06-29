import { TrendingUp, Tag, Layers, Ruler, Droplets, Eye, Info } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface AssessmentResultsProps {
  assessmentData: any;
  isProcessing: boolean;
}

export default function AssessmentResults({ assessmentData, isProcessing }: AssessmentResultsProps) {
  return (
    <Card className="mb-6">
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center">
            <TrendingUp className="text-medical-blue text-lg mr-2" />
            <h2 className="text-lg font-semibold text-gray-900">Wound Assessment Results</h2>
          </div>
          {assessmentData && (
            <div className="text-sm text-gray-500">
              <span>Case ID: {assessmentData.caseId}</span>
            </div>
          )}
        </div>

        {/* Status Banner */}
        {!assessmentData && !isProcessing && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
            <div className="flex items-center">
              <Info className="text-medical-blue mr-2" />
              <span className="text-sm text-blue-800">Upload an image to begin wound assessment</span>
            </div>
          </div>
        )}

        {/* Loading State */}
        {isProcessing && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
            <div className="flex items-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-medical-blue mr-2"></div>
              <span className="text-sm text-yellow-800">Processing wound image...</span>
            </div>
          </div>
        )}

        {/* Classification Results */}
        {assessmentData && (
          <div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center mb-2">
                  <Tag className="text-medical-blue mr-2 h-4 w-4" />
                  <span className="font-medium text-gray-900">Wound Type</span>
                </div>
                <span className="text-lg text-gray-700">{assessmentData.classification?.woundType}</span>
              </div>
              
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center mb-2">
                  <Layers className="text-medical-blue mr-2 h-4 w-4" />
                  <span className="font-medium text-gray-900">Stage</span>
                </div>
                <span className="text-lg text-gray-700">{assessmentData.classification?.stage}</span>
              </div>
              
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center mb-2">
                  <Ruler className="text-medical-blue mr-2 h-4 w-4" />
                  <span className="font-medium text-gray-900">Size</span>
                </div>
                <span className="text-lg text-gray-700 capitalize">{assessmentData.classification?.size}</span>
              </div>
              
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center mb-2">
                  <Droplets className="text-medical-blue mr-2 h-4 w-4" />
                  <span className="font-medium text-gray-900">Exudate</span>
                </div>
                <span className="text-lg text-gray-700 capitalize">{assessmentData.classification?.exudate}</span>
              </div>
            </div>

            {/* Wound Bed Condition */}
            <div className="bg-gray-50 rounded-lg p-4 mb-6">
              <div className="flex items-center mb-2">
                <Eye className="text-medical-blue mr-2 h-4 w-4" />
                <span className="font-medium text-gray-900">Wound Bed Condition</span>
              </div>
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary" className="bg-success-green text-white">
                  {assessmentData.classification?.woundBed}
                </Badge>
                {assessmentData.classification?.infectionSigns?.length > 0 && (
                  <Badge variant="secondary" className="bg-yellow-500 text-white">
                    Infection Signs Present
                  </Badge>
                )}
              </div>
            </div>

            {/* Location */}
            {assessmentData.classification?.location && (
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center mb-2">
                  <span className="font-medium text-gray-900">Location</span>
                </div>
                <span className="text-gray-700">{assessmentData.classification.location}</span>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
