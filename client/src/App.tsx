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
import NurseEvaluation from "@/pages/nurse-evaluation";
import AgentsPage from "@/pages/agents";
import NotFound from "@/pages/not-found";
import { useAuth } from "@/hooks/useAuth";

function Router() {
  const { isAuthenticated, isLoading } = useAuth();

  return (
    <Switch>
      <Route path="/">
        {() => isLoading ? null : isAuthenticated ? <MyCases /> : <Landing />}
      </Route>
      <Route path="/start-assessment" component={AuthPage} />
      <Route path="/assessment" component={Home} />
      <Route path="/care-plan/:caseId?" component={CarePlan} />
      <Route path="/my-cases" component={MyCases} />
      <Route path="/nurse-evaluation" component={NurseEvaluation} />
      <Route path="/agents" component={AgentsPage} />
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
