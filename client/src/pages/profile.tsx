import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@/hooks/useAuth";
import { useToast } from "@/hooks/use-toast";
import ProfileForm from "@/components/profile/ProfileForm";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { User, Heart, Activity, Shield, CheckCircle2, Edit } from "lucide-react";
import { isUnauthorizedError } from "@/lib/authUtils";
import type { UserProfile } from "@shared/schema";

export default function ProfilePage() {
  const [editMode, setEditMode] = useState(false);
  const { user, isAuthenticated, isLoading: authLoading } = useAuth();
  const { toast } = useToast();
  const { data: profile, isLoading: profileLoading, error } = useQuery<UserProfile>({
    queryKey: ['/api/profile'],
    enabled: isAuthenticated,
    retry: false,
  });

  // Redirect to home if not authenticated
  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      toast({
        title: "Unauthorized",
        description: "You are logged out. Logging in again...",
        variant: "destructive",
      });
      setTimeout(() => {
        window.location.href = "/api/login";
      }, 500);
      return;
    }
  }, [isAuthenticated, authLoading, toast]);

  // Handle unauthorized errors
  useEffect(() => {
    if (error && isUnauthorizedError(error)) {
      toast({
        title: "Unauthorized",
        description: "You are logged out. Logging in again...",
        variant: "destructive",
      });
      setTimeout(() => {
        window.location.href = "/api/login";
      }, 500);
      return;
    }
  }, [error, toast]);

  if (authLoading || profileLoading) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="space-y-6">
          <Skeleton className="h-8 w-64" />
          <div className="grid gap-6">
            <Skeleton className="h-64 w-full" />
            <Skeleton className="h-96 w-full" />
          </div>
        </div>
      </div>
    );
  }

  const showForm = editMode || !profile;

  const formatValue = (value: any) => {
    if (Array.isArray(value)) {
      return value.length > 0 ? value.map(v => v.replace('_', ' ')).join(', ') : 'None';
    }
    if (typeof value === 'boolean') {
      return value ? 'Yes' : 'No';
    }
    if (typeof value === 'string' && value.includes('_')) {
      return value.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
    return value || 'Not specified';
  };

  if (showForm) {
    return (
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {profile ? 'Edit Profile' : 'Complete Your Profile'}
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            {profile 
              ? 'Update your medical and personal information'
              : 'Help us provide better wound care recommendations by sharing relevant medical information'
            }
          </p>
        </div>

        <ProfileForm
          initialData={profile}
          onSuccess={() => {
            setEditMode(false);
            toast({
              title: "Profile updated",
              description: "Your profile has been successfully updated"
            });
          }}
        />

        {profile && (
          <div className="mt-6">
            <Button 
              variant="outline" 
              onClick={() => setEditMode(false)}
            >
              Cancel
            </Button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2">
            <User className="h-6 w-6" />
            My Profile
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Your medical and personal information for wound care
          </p>
        </div>
        <Button onClick={() => setEditMode(true)} className="flex items-center gap-2">
          <Edit className="h-4 w-4" />
          Edit Profile
        </Button>
      </div>

      <div className="space-y-6">
        {/* Profile Summary */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <User className="h-5 w-5" />
              Profile Summary
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Type</p>
                <Badge variant={profile.profileType === 'patient' ? 'default' : 'secondary'}>
                  {formatValue(profile.profileType)}
                </Badge>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Age</p>
                <p className="font-medium">{formatValue(profile.age)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Gender</p>
                <p className="font-medium">{formatValue(profile.gender)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Profile Status</p>
                <Badge variant="outline" className="text-green-600">
                  <CheckCircle2 className="h-3 w-3 mr-1" />
                  Complete
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Medical Conditions - Only for patients */}
        {profile.profileType === 'patient' && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Heart className="h-5 w-5" />
                Medical Conditions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Diabetes</p>
                  <p className="font-medium">{formatValue(profile.isDiabetic)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">High Blood Pressure</p>
                  <p className="font-medium">{formatValue(profile.hasHighBloodPressure)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Heart Disease</p>
                  <p className="font-medium">{formatValue(profile.hasHeartDisease)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Kidney Disease</p>
                  <p className="font-medium">{formatValue(profile.hasKidneyDisease)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Circulation Issues</p>
                  <p className="font-medium">{formatValue(profile.hasCirculationIssues)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Ostomy</p>
                  <p className="font-medium">
                    {profile.hasOstomy ? `Yes (${formatValue(profile.ostomyType)})` : 'No'}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Mobility and Activity */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Mobility & Activity
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Mobility Status</p>
                <p className="font-medium">{formatValue(profile.mobilityStatus)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Activity Level</p>
                <p className="font-medium">{formatValue(profile.activityLevel)}</p>
              </div>
              <div className="col-span-2">
                <p className="text-sm text-gray-600 dark:text-gray-400">Assistive Devices</p>
                <p className="font-medium">{formatValue(profile.assistiveDevices)}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Care Goals */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5" />
              Care Goals
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Primary Goals</p>
                <div className="flex flex-wrap gap-2 mt-2">
                  {profile.primaryGoals && profile.primaryGoals.length > 0 ? (
                    profile.primaryGoals.map((goal: string) => (
                      <Badge key={goal} variant="outline">
                        {formatValue(goal)}
                      </Badge>
                    ))
                  ) : (
                    <p className="text-gray-500 dark:text-gray-400">No goals specified</p>
                  )}
                </div>
              </div>
              {profile.carePreferences && (
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Care Preferences</p>
                  <p className="font-medium text-sm mt-1">{profile.carePreferences}</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Caregiver Information */}
        {profile.profileType === 'caregiver' && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5" />
                Caregiver Information
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Relationship to Patient</p>
                  <p className="font-medium">{formatValue(profile.relationshipToPatient)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Caregiving Experience</p>
                  <p className="font-medium">{formatValue(profile.caregivingExperience)}</p>
                </div>
                <div className="col-span-2">
                  <p className="text-sm text-gray-600 dark:text-gray-400">Professional Background</p>
                  <p className="font-medium">{formatValue(profile.professionalBackground)}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Emergency Contact */}
        {profile.emergencyContact && (
          <Card>
            <CardHeader>
              <CardTitle>Emergency Contact</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="font-medium">{profile.emergencyContact}</p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}