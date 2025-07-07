import { useLocation } from "wouter";
import { Stethoscope, CheckCircle, Users, Target, ArrowRight, Star, LogIn, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { useAuth } from "@/hooks/useAuth";

export default function Landing() {
  const [, setLocation] = useLocation();
  const { isAuthenticated, isLoading, user } = useAuth();

  return (
    <div className="min-h-screen bg-gradient-to-br from-bg-light to-gray-50">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <Stethoscope className="text-medical-blue text-2xl mr-3" />
              <span className="text-xl font-bold text-gray-900">The Wound Nurse</span>
            </div>
            <div className="flex items-center gap-3">
              {isLoading ? (
                <div className="w-8 h-8 animate-spin rounded-full border-b-2 border-medical-blue"></div>
              ) : isAuthenticated ? (
                <>
                  <Button 
                    variant="outline"
                    onClick={() => setLocation("/my-cases")}
                  >
                    <User className="mr-2 h-4 w-4" />
                    My Cases
                  </Button>
                  <Button 
                    onClick={() => setLocation("/assessment")}
                  >
                    New Assessment
                  </Button>
                  <Button 
                    variant="ghost"
                    onClick={() => {
                      localStorage.removeItem('auth_token');
                      setLocation('/');
                    }}
                  >
                    Log Out
                  </Button>
                </>
              ) : (
                <>
                  <Button 
                    variant="outline"
                    onClick={() => setLocation("/login")}
                  >
                    <LogIn className="mr-2 h-4 w-4" />
                    Log In
                  </Button>
                  
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto text-center">
          <h1 className="text-4xl sm:text-6xl font-bold text-gray-900 mb-6">
            The Wound Nurse
          </h1>
          <p className="text-lg text-gray-600 mb-12 max-w-3xl mx-auto">
            Advanced AI-powered wound assessment and care plan generation. 
            Get professional-grade wound analysis and personalized treatment recommendations in minutes.
          </p>
          
          {/* Trust Indicators */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto mb-12">
            <div className="flex flex-col items-center">
              <div className="bg-success-green/10 p-4 rounded-full mb-4">
                <Target className="h-8 w-8 text-success-green" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-2">98% Accurate</h3>
              <p className="text-gray-600">Clinical-grade accuracy powered by advanced AI models</p>
            </div>

            <div className="flex flex-col items-center">
              <div className="bg-medical-blue/10 p-4 rounded-full mb-4">
                <Users className="h-8 w-8 text-medical-blue" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-2">Trusted by Medical Professionals</h3>
              <p className="text-gray-600">Used by nurses and healthcare providers worldwide</p>
            </div>

            <div className="flex flex-col items-center">
              <div className="bg-warning-orange/10 p-4 rounded-full mb-4">
                <CheckCircle className="h-8 w-8 text-warning-orange" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-2">Evidence-Based</h3>
              <p className="text-gray-600">Recommendations based on latest clinical guidelines</p>
            </div>
          </div>
          
          {/* CTA Button */}
          <div className="flex justify-center mt-12">
            {isLoading ? (
              <div className="w-12 h-12 animate-spin rounded-full border-b-2 border-medical-blue"></div>
            ) : isAuthenticated ? (
              <Button 
                size="lg"
                onClick={() => setLocation("/my-cases")}
                className="bg-medical-blue hover:bg-blue-700 text-white px-8 py-3 text-lg font-semibold"
              >
                View My Cases
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            ) : (
              <Button 
                size="lg"
                onClick={() => setLocation("/start-assessment")}
                className="bg-medical-blue hover:bg-blue-700 text-white px-8 py-3 text-lg font-semibold"
              >
                Start New Case
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            )}
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Trusted by Healthcare Professionals</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center mb-4">
                  {[...Array(5)].map((_, i) => (
                    <Star key={i} className="h-4 w-4 text-yellow-400 fill-current" />
                  ))}
                </div>
                <p className="text-gray-600 mb-4">
                  "This tool has revolutionized how we approach wound assessment. The AI analysis is incredibly accurate and saves us valuable time."
                </p>
                <div className="text-sm">
                  <p className="font-semibold text-gray-900">Sarah Johnson, RN</p>
                  <p className="text-gray-500">Wound Care Specialist</p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center mb-4">
                  {[...Array(5)].map((_, i) => (
                    <Star key={i} className="h-4 w-4 text-yellow-400 fill-current" />
                  ))}
                </div>
                <p className="text-gray-600 mb-4">
                  "The comprehensive questionnaire ensures we capture all relevant factors. Care plans are evidence-based and practical."
                </p>
                <div className="text-sm">
                  <p className="font-semibold text-gray-900">Dr. Michael Chen</p>
                  <p className="text-gray-500">Dermatologist</p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center mb-4">
                  {[...Array(5)].map((_, i) => (
                    <Star key={i} className="h-4 w-4 text-yellow-400 fill-current" />
                  ))}
                </div>
                <p className="text-gray-600 mb-4">
                  "Perfect for both professional use and patient education. The different audience options make communication so much easier."
                </p>
                <div className="text-sm">
                  <p className="font-semibold text-gray-900">Lisa Rodriguez, NP</p>
                  <p className="text-gray-500">Nurse Practitioner</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-medical-blue">
        <div className="max-w-4xl mx-auto text-center px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl sm:text-4xl font-bold text-white mb-6">
            Ready to Get Started?
          </h2>
          <p className="text-xl text-blue-100 mb-8">
            Begin your free wound assessment today and get professional-grade care recommendations in minutes.
          </p>
          <Button 
            size="lg"
            variant="secondary"
            onClick={() => setLocation("/assessment")}
            className="bg-white text-medical-blue hover:bg-gray-100 px-8 py-3 text-lg font-semibold"
          >
            Start Your Free Assessment
            <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <div className="flex items-center justify-center mb-4">
              <Stethoscope className="text-medical-blue text-2xl mr-3" />
              <span className="text-xl font-bold">The Wound Nurse</span>
            </div>
            <p className="text-gray-400 mb-4">
              AI-powered wound care assessment for better health outcomes
            </p>
            <p className="text-sm text-gray-500">
              This tool is for educational purposes and should not replace professional medical advice.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}