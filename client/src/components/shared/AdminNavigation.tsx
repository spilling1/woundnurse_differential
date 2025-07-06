import { Button } from "@/components/ui/button";
import { Shield } from "lucide-react";
import { useLocation } from "wouter";
import { useAuth } from "@/hooks/useAuth";

export default function AdminNavigation() {
  const { user, isAuthenticated } = useAuth();
  const [, setLocation] = useLocation();

  // Only show for admin users
  if (!isAuthenticated || user?.role !== 'admin') {
    return null;
  }

  return (
    <Button
      variant="outline"
      size="sm"
      onClick={() => setLocation('/admin')}
      className="flex items-center space-x-2 bg-red-50 border-red-200 text-red-700 hover:bg-red-100 hover:border-red-300"
      title="Admin Console"
    >
      <Shield className="h-4 w-4" />
      <span>Admin</span>
    </Button>
  );
}