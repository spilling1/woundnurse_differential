import { useLocation } from "wouter";
import { Stethoscope, Circle, HelpCircle, LogOut, Settings } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/button";
import NewAssessmentFlow from "@/components/NewAssessmentFlow";

export default function Home() {
  const [, setLocation] = useLocation();
  const { isAuthenticated } = useAuth();

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
                    variant="ghost"
                    size="sm"
                    onClick={() => setLocation("/settings")}
                    className="p-2"
                    title="Settings"
                  >
                    <Settings className="h-4 w-4" />
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

      <NewAssessmentFlow />
    </div>
  );
}
