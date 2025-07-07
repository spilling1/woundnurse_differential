import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";
import { useLocation } from "wouter";
import { useMutation } from "@tanstack/react-query";
import { Lock, AlertCircle, CheckCircle } from "lucide-react";

// API function to change password
async function changePassword(data: { currentPassword: string; newPassword: string }) {
  const token = localStorage.getItem('auth_token');
  if (!token) {
    throw new Error('Not authenticated');
  }

  const response = await fetch('/api/auth/change-password', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.message || 'Failed to change password');
  }

  return response.json();
}

export default function ChangePasswordPage() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  // Change password mutation
  const changePasswordMutation = useMutation({
    mutationFn: changePassword,
    onSuccess: () => {
      toast({
        title: "Password changed successfully!",
        description: "You can now access your account normally.",
      });
      setLocation('/my-cases');
    },
    onError: (error: Error) => {
      toast({
        title: "Failed to change password",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const handleChangePassword = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!currentPassword || !newPassword || !confirmPassword) {
      toast({
        title: "Missing information",
        description: "Please fill in all fields",
        variant: "destructive",
      });
      return;
    }

    if (newPassword !== confirmPassword) {
      toast({
        title: "Password mismatch",
        description: "New passwords do not match",
        variant: "destructive",
      });
      return;
    }

    if (newPassword.length < 8) {
      toast({
        title: "Password too short",
        description: "New password must be at least 8 characters long",
        variant: "destructive",
      });
      return;
    }

    if (newPassword === currentPassword) {
      toast({
        title: "Same password",
        description: "New password must be different from your current password",
        variant: "destructive",
      });
      return;
    }

    changePasswordMutation.mutate({
      currentPassword,
      newPassword,
    });
  };

  const handleSkip = () => {
    // For now, allow skipping but warn the user
    toast({
      title: "Password change skipped",
      description: "You should change your password soon for security",
      variant: "destructive",
    });
    setLocation('/my-cases');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-blue to-medical-teal flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo/Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Wound Nurses</h1>
          <p className="text-blue-100">Secure Your Account</p>
        </div>

        <Card className="bg-white/95 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-center text-medical-blue flex items-center justify-center">
              <Lock className="mr-2 h-5 w-5" />
              Change Password Required
            </CardTitle>
            <CardDescription className="text-center">
              For security reasons, you must change your default password before continuing.
            </CardDescription>
          </CardHeader>
          
          <CardContent className="space-y-6">
            {/* Security Alert */}
            <Alert className="bg-amber-50 border-amber-200">
              <AlertCircle className="h-4 w-4 text-amber-600" />
              <AlertDescription className="text-amber-800">
                <strong>Security Notice:</strong> Your account was set up with a temporary password. Please create a secure password to protect your medical information.
              </AlertDescription>
            </Alert>

            <form onSubmit={handleChangePassword} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="current-password">Current Password</Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                  <Input
                    id="current-password"
                    type="password"
                    placeholder="Enter current password (Woundnurse)"
                    value={currentPassword}
                    onChange={(e) => setCurrentPassword(e.target.value)}
                    className="pl-10"
                    required
                  />
                </div>
                <p className="text-sm text-gray-600">
                  Hint: Your current password is "Woundnurse"
                </p>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="new-password">New Password</Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                  <Input
                    id="new-password"
                    type="password"
                    placeholder="Create a strong password (min 8 characters)"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    className="pl-10"
                    required
                    minLength={8}
                  />
                </div>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="confirm-new-password">Confirm New Password</Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                  <Input
                    id="confirm-new-password"
                    type="password"
                    placeholder="Confirm your new password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className="pl-10"
                    required
                  />
                </div>
              </div>

              {/* Password requirements */}
              <Alert className="bg-blue-50 border-blue-200">
                <CheckCircle className="h-4 w-4 text-blue-600" />
                <AlertDescription className="text-blue-800">
                  <strong>Password Requirements:</strong>
                  <ul className="mt-1 text-sm list-disc list-inside space-y-1">
                    <li>At least 8 characters long</li>
                    <li>Different from your current password</li>
                    <li>Should include a mix of letters, numbers, and symbols for best security</li>
                  </ul>
                </AlertDescription>
              </Alert>

              <div className="space-y-3">
                <Button 
                  type="submit" 
                  className="w-full bg-medical-blue hover:bg-medical-blue/90"
                  disabled={changePasswordMutation.isPending}
                >
                  {changePasswordMutation.isPending ? (
                    "Changing password..."
                  ) : (
                    <>
                      <Lock className="mr-2 h-4 w-4" />
                      Change Password
                    </>
                  )}
                </Button>

                <Button 
                  type="button"
                  variant="ghost"
                  className="w-full text-gray-600 hover:text-gray-800"
                  onClick={handleSkip}
                >
                  Skip for now (not recommended)
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}