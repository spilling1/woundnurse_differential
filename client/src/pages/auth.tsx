import { useEffect } from "react";
import { Stethoscope, Shield, FileText, Users, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useAuth } from "@/hooks/useAuth";
import { useLocation } from "wouter";

export default function AuthPage() {
  const { isAuthenticated, isLoading } = useAuth();
  const [, setLocation] = useLocation();

  useEffect(() => {
    if (isAuthenticated) {
      setLocation("/assessment");
    }
  }, [isAuthenticated, setLocation]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-medical-blue to-medical-teal flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-white"></div>
      </div>
    );
  }

  if (isAuthenticated) {
    return null; // Redirecting
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-blue to-medical-teal">
      {/* Header */}
      <header className="border-b border-white/20 bg-white/10 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <Stethoscope className="text-white text-2xl mr-3" />
              <span className="text-xl font-bold text-white">Wound Nurses</span>
            </div>
            <Button 
              variant="ghost" 
              className="text-white hover:bg-white/20"
              onClick={() => setLocation("/")}
            >
              Back to Home
            </Button>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto">
          {/* Main Content */}
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold text-white mb-4">
              Start Your Free Wound Assessment
            </h1>
            <p className="text-xl text-medical-light mb-8">
              Create an account to save your assessments and track your wound care progress over time
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
            {/* Benefits Card */}
            <Card className="bg-white/95 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center text-medical-blue">
                  <Shield className="mr-2 h-5 w-5" />
                  Why Create an Account?
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-start">
                  <FileText className="mr-3 h-5 w-5 text-medical-blue mt-0.5" />
                  <div>
                    <p className="font-medium text-gray-900">Save Your Assessments</p>
                    <p className="text-sm text-gray-600">Access your wound care history anytime</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <Users className="mr-3 h-5 w-5 text-medical-blue mt-0.5" />
                  <div>
                    <p className="font-medium text-gray-900">Track Progress</p>
                    <p className="text-sm text-gray-600">Monitor healing over time with photo comparisons</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <Shield className="mr-3 h-5 w-5 text-medical-blue mt-0.5" />
                  <div>
                    <p className="font-medium text-gray-900">Secure & Private</p>
                    <p className="text-sm text-gray-600">HIPAA-compliant data protection</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Login Card */}
            <Card className="bg-white/95 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-center text-medical-blue">
                  Sign Up or Log In
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="text-center">
                  <p className="text-gray-600 mb-6">
                    Use your Replit account to get started quickly and securely
                  </p>
                  
                  <Button 
                    size="lg"
                    className="w-full bg-medical-blue hover:bg-medical-blue/90 text-lg py-6"
                    onClick={() => window.location.href = "/api/login"}
                  >
                    <ArrowRight className="mr-2 h-5 w-5" />
                    Continue with Replit
                  </Button>
                </div>

                <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <span className="w-full border-t border-gray-300" />
                  </div>
                  <div className="relative flex justify-center text-xs uppercase">
                    <span className="bg-white px-2 text-gray-500">Or</span>
                  </div>
                </div>

                <div className="text-center">
                  <p className="text-sm text-gray-600 mb-4">
                    Want to try without an account?
                  </p>
                  <Button 
                    variant="outline"
                    className="w-full"
                    onClick={() => setLocation("/assessment")}
                  >
                    Continue as Guest
                  </Button>
                  <p className="text-xs text-gray-500 mt-2">
                    Note: Guest assessments are not saved
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Trust Indicators */}
          <div className="text-center">
            <p className="text-medical-light text-sm mb-4">
              Trusted by healthcare professionals and patients worldwide
            </p>
            <div className="flex justify-center items-center space-x-8 text-white/70">
              <div className="flex items-center">
                <Shield className="h-4 w-4 mr-2" />
                <span className="text-sm">HIPAA Compliant</span>
              </div>
              <div className="flex items-center">
                <FileText className="h-4 w-4 mr-2" />
                <span className="text-sm">Evidence-Based</span>
              </div>
              <div className="flex items-center">
                <Users className="h-4 w-4 mr-2" />
                <span className="text-sm">Clinically Validated</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}