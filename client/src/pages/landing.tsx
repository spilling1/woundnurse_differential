import { useLocation } from "wouter";
import { Stethoscope, CheckCircle, Users, Target, ArrowRight, Star } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export default function Landing() {
  const [, setLocation] = useLocation();

  return (
    <div className="min-h-screen bg-gradient-to-br from-bg-light to-gray-50">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <Stethoscope className="text-medical-blue text-2xl mr-3" />
              <span className="text-xl font-bold text-gray-900">Wound Nurses</span>
            </div>
            <Button 
              variant="outline"
              onClick={() => setLocation("/assessment")}
            >
              Start Assessment
            </Button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto text-center">
          <h1 className="text-4xl sm:text-6xl font-bold text-gray-900 mb-6">
            Wound Nurses
          </h1>
          <p className="text-xl sm:text-2xl text-medical-blue font-semibold mb-4">
            Powered by AI
          </p>
          <p className="text-lg text-gray-600 mb-8 max-w-3xl mx-auto">
            Advanced AI-powered wound assessment and care plan generation. 
            Get professional-grade wound analysis and personalized treatment recommendations in minutes.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12">
            <Button 
              size="lg"
              onClick={() => setLocation("/assessment")}
              className="bg-medical-blue hover:bg-blue-700 text-white px-8 py-3 text-lg font-semibold"
            >
              Start Your Assessment for Free
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </div>

          {/* Trust Indicators */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto">
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
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">How It Works</h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Get professional wound assessment and personalized care plans in three simple steps
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <Card>
              <CardContent className="p-6 text-center">
                <div className="bg-medical-blue/10 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl font-bold text-medical-blue">1</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Upload & Answer</h3>
                <p className="text-gray-600">
                  Upload wound images and answer comprehensive health questions for accurate assessment
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6 text-center">
                <div className="bg-medical-blue/10 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl font-bold text-medical-blue">2</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">AI Analysis</h3>
                <p className="text-gray-600">
                  Advanced AI analyzes wound characteristics, considers your health context, and generates insights
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6 text-center">
                <div className="bg-medical-blue/10 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-2xl font-bold text-medical-blue">3</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Get Care Plan</h3>
                <p className="text-gray-600">
                  Receive personalized treatment recommendations tailored to your specific needs and situation
                </p>
              </CardContent>
            </Card>
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
              <span className="text-xl font-bold">Wound Nurses</span>
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