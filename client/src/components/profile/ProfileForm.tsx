import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { insertUserProfileSchema, type InsertUserProfile } from "@shared/schema";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Textarea } from "@/components/ui/textarea";
import { Heart, Shield, Activity, Users, CheckCircle2 } from "lucide-react";

interface ProfileFormProps {
  initialData?: InsertUserProfile;
  onSuccess?: () => void;
}

const profileTypeOptions = [
  { value: "patient", label: "Patient", description: "I am receiving wound care" },
  { value: "caregiver", label: "Caregiver", description: "I am caring for someone with wounds" }
];

const goalOptions = [
  "quality_of_life",
  "pain_management", 
  "independence",
  "mobility",
  "prevent_infection",
  "healing",
  "return_to_activities"
];

export default function ProfileForm({ initialData, onSuccess }: ProfileFormProps) {
  const [selectedGoals, setSelectedGoals] = useState<string[]>(initialData?.primaryGoals || []);
  const [selectedDevices, setSelectedDevices] = useState<string[]>(initialData?.assistiveDevices || []);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const form = useForm<InsertUserProfile>({
    resolver: zodResolver(insertUserProfileSchema),
    defaultValues: {
      profileType: initialData?.profileType || "patient",
      age: initialData?.age || undefined,
      gender: initialData?.gender || "",
      isDiabetic: initialData?.isDiabetic ?? false,
      hasHighBloodPressure: initialData?.hasHighBloodPressure ?? false,
      hasHeartDisease: initialData?.hasHeartDisease ?? false,
      hasKidneyDisease: initialData?.hasKidneyDisease ?? false,
      hasCirculationIssues: initialData?.hasCirculationIssues ?? false,
      hasOstomy: initialData?.hasOstomy ?? false,
      ostomyType: initialData?.ostomyType || "",
      mobilityStatus: initialData?.mobilityStatus || "fully_mobile",
      assistiveDevices: initialData?.assistiveDevices || [],
      activityLevel: initialData?.activityLevel || "moderately_active",
      currentMedications: initialData?.currentMedications || "",
      medicationAllergies: initialData?.medicationAllergies || "",
      otherAllergies: initialData?.otherAllergies || "",
      nutritionStatus: initialData?.nutritionStatus || "good",
      dietRestrictions: initialData?.dietRestrictions || "",
      smokingStatus: initialData?.smokingStatus || "never",
      alcoholUse: initialData?.alcoholUse || "none",
      primaryGoals: initialData?.primaryGoals || [],
      carePreferences: initialData?.carePreferences || "",
      emergencyContact: initialData?.emergencyContact || "",
      relationshipToPatient: initialData?.relationshipToPatient || "",
      caregivingExperience: initialData?.caregivingExperience || "none",
      professionalBackground: initialData?.professionalBackground || "non_medical",
      profileCompleted: true
    }
  });

  const mutation = useMutation({
    mutationFn: async (data: InsertUserProfile) => {
      return await apiRequest("/api/profile", "POST", data);
    },
    onSuccess: () => {
      toast({
        title: "Profile saved successfully",
        description: "Your profile information has been updated"
      });
      queryClient.invalidateQueries({ queryKey: ['/api/profile'] });
      onSuccess?.();
    },
    onError: (error: any) => {
      toast({
        title: "Error saving profile",
        description: error.message || "Failed to save profile",
        variant: "destructive"
      });
    }
  });

  const onSubmit = (data: InsertUserProfile) => {
    const formData = {
      ...data,
      primaryGoals: selectedGoals,
      assistiveDevices: selectedDevices
    };
    mutation.mutate(formData);
  };

  const profileType = form.watch("profileType");
  const hasOstomy = form.watch("hasOstomy");

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
        
        {/* Profile Type Selection */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" />
              Profile Type
            </CardTitle>
            <CardDescription>
              Select whether you are a patient or caregiver
            </CardDescription>
          </CardHeader>
          <CardContent>
            <FormField
              control={form.control}
              name="profileType"
              render={({ field }) => (
                <FormItem>
                  <Select onValueChange={field.onChange} defaultValue={field.value}>
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue placeholder="Select profile type" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      {profileTypeOptions.map((option) => (
                        <SelectItem key={option.value} value={option.value}>
                          <div className="flex flex-col">
                            <span className="font-medium">{option.label}</span>
                            <span className="text-sm text-muted-foreground">{option.description}</span>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <FormMessage />
                </FormItem>
              )}
            />
          </CardContent>
        </Card>

        {/* Basic Information */}
        <Card>
          <CardHeader>
            <CardTitle>Basic Information</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <FormField
                control={form.control}
                name="age"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Age</FormLabel>
                    <FormControl>
                      <Input type="number" placeholder="Enter age" value={field.value || ""} onChange={(e) => field.onChange(e.target.value ? parseInt(e.target.value) : undefined)} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="gender"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Gender</FormLabel>
                    <Select onValueChange={field.onChange} defaultValue={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Select gender" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="male">Male</SelectItem>
                        <SelectItem value="female">Female</SelectItem>
                        <SelectItem value="other">Other</SelectItem>
                        <SelectItem value="prefer_not_to_say">Prefer not to say</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>
          </CardContent>
        </Card>

        {/* Medical Conditions - Only for patients */}
        {profileType === "patient" && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Heart className="h-5 w-5" />
                Medical Conditions
              </CardTitle>
              <CardDescription>
                Select any medical conditions that apply to you
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <FormField
                  control={form.control}
                  name="isDiabetic"
                  render={({ field }) => (
                    <FormItem className="flex flex-row items-start space-x-3 space-y-0">
                      <FormControl>
                        <Checkbox
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                      </FormControl>
                      <div className="space-y-1 leading-none">
                        <FormLabel>Diabetes</FormLabel>
                        <FormDescription>
                          Type 1 or Type 2 diabetes
                        </FormDescription>
                      </div>
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="hasHighBloodPressure"
                  render={({ field }) => (
                    <FormItem className="flex flex-row items-start space-x-3 space-y-0">
                      <FormControl>
                        <Checkbox
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                      </FormControl>
                      <div className="space-y-1 leading-none">
                        <FormLabel>High Blood Pressure</FormLabel>
                        <FormDescription>
                          Hypertension
                        </FormDescription>
                      </div>
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="hasHeartDisease"
                  render={({ field }) => (
                    <FormItem className="flex flex-row items-start space-x-3 space-y-0">
                      <FormControl>
                        <Checkbox
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                      </FormControl>
                      <div className="space-y-1 leading-none">
                        <FormLabel>Heart Disease</FormLabel>
                        <FormDescription>
                          Cardiovascular conditions
                        </FormDescription>
                      </div>
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="hasKidneyDisease"
                  render={({ field }) => (
                    <FormItem className="flex flex-row items-start space-x-3 space-y-0">
                      <FormControl>
                        <Checkbox
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                      </FormControl>
                      <div className="space-y-1 leading-none">
                        <FormLabel>Kidney Disease</FormLabel>
                        <FormDescription>
                          Renal conditions
                        </FormDescription>
                      </div>
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="hasCirculationIssues"
                  render={({ field }) => (
                    <FormItem className="flex flex-row items-start space-x-3 space-y-0">
                      <FormControl>
                        <Checkbox
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                      </FormControl>
                      <div className="space-y-1 leading-none">
                        <FormLabel>Circulation Issues</FormLabel>
                        <FormDescription>
                          Poor blood flow
                        </FormDescription>
                      </div>
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="hasOstomy"
                  render={({ field }) => (
                    <FormItem className="flex flex-row items-start space-x-3 space-y-0">
                      <FormControl>
                        <Checkbox
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                      </FormControl>
                      <div className="space-y-1 leading-none">
                        <FormLabel>Ostomy</FormLabel>
                        <FormDescription>
                          Colostomy, ileostomy, or urostomy
                        </FormDescription>
                      </div>
                    </FormItem>
                  )}
                />
              </div>
              
              {hasOstomy && (
                <FormField
                  control={form.control}
                  name="ostomyType"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Ostomy Type</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select ostomy type" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="colostomy">Colostomy</SelectItem>
                          <SelectItem value="ileostomy">Ileostomy</SelectItem>
                          <SelectItem value="urostomy">Urostomy</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              )}
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
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <FormField
                control={form.control}
                name="mobilityStatus"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Mobility Status</FormLabel>
                    <Select onValueChange={field.onChange} defaultValue={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="fully_mobile">Fully Mobile</SelectItem>
                        <SelectItem value="limited_mobility">Limited Mobility</SelectItem>
                        <SelectItem value="wheelchair">Wheelchair</SelectItem>
                        <SelectItem value="bed_ridden">Bed Ridden</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="activityLevel"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Activity Level</FormLabel>
                    <Select onValueChange={field.onChange} defaultValue={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="very_active">Very Active</SelectItem>
                        <SelectItem value="moderately_active">Moderately Active</SelectItem>
                        <SelectItem value="sedentary">Sedentary</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>
            
            <div>
              <FormLabel>Assistive Devices</FormLabel>
              <div className="grid grid-cols-3 gap-2 mt-2">
                {["walker", "crutches", "cane", "wheelchair", "mobility_scooter", "prosthetic"].map((device) => (
                  <div key={device} className="flex items-center space-x-2">
                    <Checkbox
                      id={device}
                      checked={selectedDevices.includes(device)}
                      onCheckedChange={(checked) => {
                        if (checked) {
                          setSelectedDevices([...selectedDevices, device]);
                        } else {
                          setSelectedDevices(selectedDevices.filter(d => d !== device));
                        }
                      }}
                    />
                    <label htmlFor={device} className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 capitalize">
                      {device.replace('_', ' ')}
                    </label>
                  </div>
                ))}
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
            <CardDescription>
              Select your primary goals for wound care
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-2">
              {goalOptions.map((goal) => (
                <div key={goal} className="flex items-center space-x-2">
                  <Checkbox
                    id={goal}
                    checked={selectedGoals.includes(goal)}
                    onCheckedChange={(checked) => {
                      if (checked) {
                        setSelectedGoals([...selectedGoals, goal]);
                      } else {
                        setSelectedGoals(selectedGoals.filter(g => g !== goal));
                      }
                    }}
                  />
                  <label htmlFor={goal} className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 capitalize">
                    {goal.replace('_', ' ')}
                  </label>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Caregiver Information */}
        {profileType === "caregiver" && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5" />
                Caregiver Information
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <FormField
                  control={form.control}
                  name="relationshipToPatient"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Relationship to Patient</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select relationship" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="spouse">Spouse/Partner</SelectItem>
                          <SelectItem value="parent">Parent</SelectItem>
                          <SelectItem value="child">Child</SelectItem>
                          <SelectItem value="sibling">Sibling</SelectItem>
                          <SelectItem value="friend">Friend</SelectItem>
                          <SelectItem value="professional">Professional Caregiver</SelectItem>
                          <SelectItem value="other">Other</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="caregivingExperience"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Caregiving Experience</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="none">No Experience</SelectItem>
                          <SelectItem value="some">Some Experience</SelectItem>
                          <SelectItem value="experienced">Very Experienced</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
              <FormField
                control={form.control}
                name="professionalBackground"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Professional Background</FormLabel>
                    <Select onValueChange={field.onChange} defaultValue={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="non_medical">Non-Medical</SelectItem>
                        <SelectItem value="medical">Medical Professional</SelectItem>
                        <SelectItem value="retired_medical">Retired Medical Professional</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </CardContent>
          </Card>
        )}

        {/* Additional Information */}
        <Card>
          <CardHeader>
            <CardTitle>Additional Information</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <FormField
              control={form.control}
              name="carePreferences"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Care Preferences</FormLabel>
                  <FormControl>
                    <Textarea
                      placeholder="Describe any specific care preferences or concerns..."
                      className="resize-none"
                      {...field}
                    />
                  </FormControl>
                  <FormDescription>
                    Optional: Share any specific needs or preferences for wound care
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="emergencyContact"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Emergency Contact</FormLabel>
                  <FormControl>
                    <Input placeholder="Name and phone number" {...field} />
                  </FormControl>
                  <FormDescription>
                    Optional: Emergency contact information
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
          </CardContent>
        </Card>

        <div className="flex gap-4">
          <Button
            type="submit"
            className="flex-1"
            disabled={mutation.isPending}
          >
            {mutation.isPending ? "Saving..." : "Save Profile"}
          </Button>
        </div>
      </form>
    </Form>
  );
}