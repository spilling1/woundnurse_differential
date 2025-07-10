import { useLocation, useParams } from "wouter";
import { ArrowLeft, AlertTriangle, Camera, RefreshCw, Home } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useState, useEffect } from "react";

export default function UnsupportedWound() {
  const [, setLocation] = useLocation();
  const params = useParams();
  
  // Get wound type and confidence from URL params or query string
  const urlParams = new URLSearchParams(window.location.search);
  const woundType = urlParams.get('woundType') || params.woundType || 'Unknown';
  const confidence = urlParams.get('confidence') || '85';
  const reasoning = urlParams.get('reasoning') || 'Visual analysis performed by AI';
  const message = urlParams.get('message') || 'This wound type is not currently supported';
  const supportedTypesParam = urlParams.get('supportedTypes') || '';
  const supportedTypes = supportedTypesParam ? supportedTypesParam.split('|') : [];
  const caseId = urlParams.get('caseId') || params.caseId;

  // Removed auto-redirect timer - users will use action buttons instead

  const handleResubmit = () => {
    setLocation('/assessment');
  };

  const handleClose = () => {
    setLocation('/my-cases');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="w-full max-w-2xl">
        {/* Main Card */}
        <Card className="shadow-2xl border-0 overflow-hidden">
          <CardHeader className="bg-gradient-to-r from-amber-500 to-orange-500 text-white text-center py-8">
            <div className="flex justify-center mb-4">
              <div className="bg-white/20 rounded-full p-4">
                <AlertTriangle className="h-12 w-12 text-white" />
              </div>
            </div>
            <CardTitle className="text-2xl font-bold">
              Wound Type Not Currently Supported
            </CardTitle>
          </CardHeader>
          
          <CardContent className="p-8 text-center space-y-6">
            {/* Analysis Results */}
            <div className="bg-gray-50 rounded-lg p-6 space-y-4">
              <h3 className="text-lg font-semibold text-gray-900">Analysis Results</h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Detected Wound Type:</span>
                  <Badge variant="outline" className="text-base px-3 py-1">
                    {woundType}
                  </Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">AI Confidence:</span>
                  <Badge 
                    variant={parseInt(confidence) >= 80 ? "default" : "secondary"}
                    className="text-base px-3 py-1"
                  >
                    {confidence}%
                  </Badge>
                </div>
              </div>
            </div>

            {/* Main Message */}
            <div className="space-y-4 text-gray-700">
              <p className="text-lg leading-relaxed font-medium">
                {message}
              </p>
              <p className="leading-relaxed">
                We have <strong>{confidence}% confidence</strong> in this assessment because of {reasoning}.
              </p>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-blue-800">
                <p className="font-medium">If you disagree with this assessment:</p>
                <p className="mt-2 text-sm">
                  Please resubmit your request with additional pictures from different angles, with better lighting, or showing more detail of the wound area.
                </p>
              </div>
            </div>

            {/* Supported Types */}
            <div className="bg-gray-50 rounded-lg p-6 text-left">
              <h4 className="font-semibold text-gray-900 mb-3 text-center">Currently Supported Wound Types:</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm text-gray-700">
                {supportedTypes.length > 0 ? (
                  supportedTypes.map((type, index) => (
                    <div key={index}>• {type}</div>
                  ))
                ) : (
                  // Default supported types if none provided
                  <>
                    <div>• Pressure Injuries</div>
                    <div>• Venous Ulcers</div>
                    <div>• Arterial Insufficiency Ulcers</div>
                    <div>• Diabetic Ulcers</div>
                    <div>• Surgical Wounds</div>
                    <div>• Traumatic Wounds</div>
                    <div>• Ischemic Wounds</div>
                    <div>• Radiation Wounds</div>
                    <div>• Infectious Wounds</div>
                  </>
                )}
              </div>
            </div>



            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 pt-6">
              <Button 
                onClick={handleResubmit}
                className="flex-1 bg-blue-600 hover:bg-blue-700 text-white py-3"
                size="lg"
              >
                <RefreshCw className="mr-2 h-5 w-5" />
                Resubmit
              </Button>
              <Button 
                onClick={handleClose}
                variant="outline"
                className="flex-1 py-3"
                size="lg"
              >
                <ArrowLeft className="mr-2 h-5 w-5" />
                Close
              </Button>
            </div>


          </CardContent>
        </Card>

        {/* Back Button */}
        <div className="text-center mt-6">
          <Button 
            variant="ghost" 
            onClick={() => setLocation('/')}
            className="text-gray-600 hover:text-gray-800"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Home
          </Button>
        </div>
      </div>
    </div>
  );
}