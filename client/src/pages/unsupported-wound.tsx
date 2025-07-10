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
  const caseId = urlParams.get('caseId') || params.caseId;

  const [countdown, setCountdown] = useState(30);

  // Countdown timer for auto-redirect
  useEffect(() => {
    const timer = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          setLocation('/');
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [setLocation]);

  const handleTryAgain = () => {
    setLocation('/');
  };

  const handleGoHome = () => {
    setLocation('/');
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
              <p className="text-lg leading-relaxed">
                Our AI analysis indicates this appears to be a <strong>{woundType.toLowerCase()}</strong> with {confidence}% confidence.
              </p>
              <p className="leading-relaxed">
                The Wound Nurses system does not currently support analysis and treatment recommendations for this type of wound.
              </p>
            </div>

            {/* Supported Types */}
            <div className="bg-blue-50 rounded-lg p-6 text-left">
              <h4 className="font-semibold text-blue-900 mb-3 text-center">Currently Supported Wound Types:</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm text-blue-800">
                <div>• Venous Ulcers</div>
                <div>• Arterial Insufficiency Ulcers</div>
                <div>• Diabetic Ulcers</div>
                <div>• Surgical Wounds</div>
                <div>• Traumatic Wounds</div>
                <div>• Ischemic Wounds</div>
                <div>• Radiation Wounds</div>
                <div>• Infectious Wounds</div>
              </div>
            </div>

            {/* What to do next */}
            <div className="bg-green-50 rounded-lg p-6 space-y-3">
              <h4 className="font-semibold text-green-900">What You Can Do:</h4>
              <div className="text-left space-y-2 text-green-800">
                <div className="flex items-start space-x-2">
                  <Camera className="h-5 w-5 mt-0.5 flex-shrink-0" />
                  <span>Try uploading additional images from different angles for better analysis</span>
                </div>
                <div className="flex items-start space-x-2">
                  <AlertTriangle className="h-5 w-5 mt-0.5 flex-shrink-0" />
                  <span>Consult with a healthcare professional for proper assessment</span>
                </div>
                <div className="flex items-start space-x-2">
                  <RefreshCw className="h-5 w-5 mt-0.5 flex-shrink-0" />
                  <span>Contact your wound care specialist or primary care provider</span>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 pt-6">
              <Button 
                onClick={handleTryAgain}
                className="flex-1 bg-blue-600 hover:bg-blue-700 text-white py-3"
                size="lg"
              >
                <Camera className="mr-2 h-5 w-5" />
                Try Again with Different Images
              </Button>
              <Button 
                onClick={handleGoHome}
                variant="outline"
                className="flex-1 py-3"
                size="lg"
              >
                <Home className="mr-2 h-5 w-5" />
                Go to Home Page
              </Button>
            </div>

            {/* Auto-redirect notice */}
            <div className="text-sm text-gray-500 mt-6">
              You will be automatically redirected to the home page in {countdown} seconds
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