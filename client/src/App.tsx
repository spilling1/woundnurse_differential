import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import Landing from "@/pages/landing";
import AuthPage from "@/pages/auth";
import Home from "@/pages/home";
import CarePlan from "@/pages/care-plan";
import MyCases from "@/pages/my-cases";
import FollowUp from "@/pages/follow-up";
import NurseEvaluation from "@/pages/nurse-evaluation";
import AgentsPage from "@/pages/agents";
import SettingsPage from "@/pages/settings";
import AdminDashboard from "@/pages/admin-dashboard";
import NotFound from "@/pages/not-found";
import { useAuth } from "@/hooks/useAuth";

function Router() {
  // Temporarily disable auth to fix infinite loop - will re-enable with proper flow
  return (
    <Switch>
      <Route path="/" component={Landing} />
      <Route path="/start-assessment" component={AuthPage} />
      <Route path="/assessment" component={Home} />
      <Route path="/care-plan/:caseId?" component={CarePlan} />
      <Route path="/my-cases" component={MyCases} />
      <Route path="/follow-up/:caseId" component={FollowUp} />
      <Route path="/nurse-evaluation/:caseId" component={NurseEvaluation} />
      <Route path="/agents" component={AgentsPage} />
      <Route path="/settings" component={SettingsPage} />
      <Route path="/admin" component={AdminDashboard} />
      <Route path="/admin-dashboard" component={AdminDashboard} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Router />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
