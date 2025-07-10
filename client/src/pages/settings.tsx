import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Settings, Save, RefreshCw, ArrowLeft, Plus, Edit, Trash2, Eye, EyeOff } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useLocation } from "wouter";
import { useAuth } from "@/hooks/useAuth";
import { apiRequest } from "@/lib/queryClient";

function SettingsPage() {
  const [systemPrompts, setSystemPrompts] = useState("");
  const [carePlanStructure, setCarePlanStructure] = useState("");
  const [specificWoundCare, setSpecificWoundCare] = useState("");
  const [questionsGuidelines, setQuestionsGuidelines] = useState("");
  const [productRecommendations, setProductRecommendations] = useState("");
  
  // Wound type management state
  const [selectedWoundType, setSelectedWoundType] = useState("");
  const [woundTypeInstructions, setWoundTypeInstructions] = useState("");
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingWoundType, setEditingWoundType] = useState<any>(null);
  const [newWoundType, setNewWoundType] = useState({
    name: "",
    displayName: "",
    description: "",
    instructions: "",
    isEnabled: true,
    priority: 50
  });
  
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const { user, isAuthenticated, isLoading: authLoading } = useAuth();
  const [, setLocation] = useLocation();

  // Query to fetch current AI instructions
  const { data: agentData, isLoading } = useQuery({
    queryKey: ["/api/agents"],
  });

  // Query to fetch wound types
  const { data: woundTypes, isLoading: woundTypesLoading } = useQuery({
    queryKey: ["/api/admin/wound-types"],
    enabled: isAuthenticated && user?.role === 'admin',
  });

  // Update textareas when data is loaded
  useEffect(() => {
    if (agentData && typeof agentData === 'object') {
      const data = agentData as any;
      setSystemPrompts(data.systemPrompts || "");
      setCarePlanStructure(data.carePlanStructure || "");
      setSpecificWoundCare(data.specificWoundCare || "");
      setQuestionsGuidelines(data.questionsGuidelines || "");
      setProductRecommendations(data.productRecommendations || "");
    }
  }, [agentData]);

  // Update wound type instructions when selection changes
  useEffect(() => {
    if (selectedWoundType && woundTypes) {
      const selectedType = (woundTypes as any[]).find(type => type.id.toString() === selectedWoundType);
      if (selectedType) {
        setWoundTypeInstructions(selectedType.instructions || "");
      }
    }
  }, [selectedWoundType, woundTypes]);

  // Mutation to update AI instructions
  const updateMutation = useMutation({
    mutationFn: async (newInstructions: {
      systemPrompts: string;
      carePlanStructure: string;
      specificWoundCare: string;
      questionsGuidelines: string;
      productRecommendations: string;
    }) => {
      const response = await fetch('/api/agents', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newInstructions),
      });
      if (!response.ok) {
        throw new Error('Failed to update instructions');
      }
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Settings Updated",
        description: "AI configuration has been successfully updated.",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/agents"] });
    },
    onError: (error: Error) => {
      toast({
        title: "Update Failed",
        description: error.message || "Failed to update AI configuration.",
        variant: "destructive",
      });
    },
  });

  // Wound type mutations
  const createWoundTypeMutation = useMutation({
    mutationFn: async (woundType: any) => {
      const response = await apiRequest('POST', '/api/admin/wound-types', woundType);
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Wound Type Created",
        description: "New wound type has been successfully created.",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/admin/wound-types"] });
      setDialogOpen(false);
      setNewWoundType({
        name: "",
        displayName: "",
        description: "",
        instructions: "",
        isEnabled: true,
        priority: 50
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Creation Failed",
        description: error.message || "Failed to create wound type.",
        variant: "destructive",
      });
    },
  });

  const updateWoundTypeMutation = useMutation({
    mutationFn: async ({ id, ...updates }: any) => {
      const response = await apiRequest('PATCH', `/api/admin/wound-types/${id}`, updates);
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Wound Type Updated",
        description: "Wound type has been successfully updated.",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/admin/wound-types"] });
    },
    onError: (error: Error) => {
      toast({
        title: "Update Failed",
        description: error.message || "Failed to update wound type.",
        variant: "destructive",
      });
    },
  });

  const deleteWoundTypeMutation = useMutation({
    mutationFn: async (id: number) => {
      const response = await apiRequest('DELETE', `/api/admin/wound-types/${id}`);
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Wound Type Deleted",
        description: "Wound type has been successfully deleted.",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/admin/wound-types"] });
    },
    onError: (error: Error) => {
      toast({
        title: "Deletion Failed",
        description: error.message || "Failed to delete wound type.",
        variant: "destructive",
      });
    },
  });

  const toggleWoundTypeMutation = useMutation({
    mutationFn: async ({ id, enabled }: { id: number; enabled: boolean }) => {
      const response = await apiRequest('PATCH', `/api/admin/wound-types/${id}/toggle`, { enabled });
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Wound Type Updated",
        description: "Wound type status has been updated.",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/admin/wound-types"] });
    },
    onError: (error: Error) => {
      toast({
        title: "Update Failed",
        description: error.message || "Failed to update wound type status.",
        variant: "destructive",
      });
    },
  });

  const handleSave = () => {
    updateMutation.mutate({
      systemPrompts,
      carePlanStructure,
      specificWoundCare,
      questionsGuidelines,
      productRecommendations
    });
  };

  const handleReset = () => {
    if (agentData && typeof agentData === 'object') {
      const data = agentData as any;
      setSystemPrompts(data.systemPrompts || "");
      setCarePlanStructure(data.carePlanStructure || "");
      setSpecificWoundCare(data.specificWoundCare || "");
      setQuestionsGuidelines(data.questionsGuidelines || "");
      setProductRecommendations(data.productRecommendations || "");
      toast({
        title: "Reset Complete",
        description: "Settings have been reset to saved values.",
      });
    }
  };

  // Wound type handler functions
  const handleWoundTypeChange = (value: string) => {
    setSelectedWoundType(value);
  };

  const handleWoundTypeInstructionsChange = (value: string) => {
    setWoundTypeInstructions(value);
  };

  const handleSaveWoundType = () => {
    if (selectedWoundType && woundTypeInstructions) {
      updateWoundTypeMutation.mutate({
        id: parseInt(selectedWoundType),
        instructions: woundTypeInstructions
      });
    }
  };

  const handleCreateWoundType = () => {
    if (newWoundType.name && newWoundType.displayName && newWoundType.instructions) {
      createWoundTypeMutation.mutate({
        name: newWoundType.name.toLowerCase().replace(/\s+/g, '_'),
        displayName: newWoundType.displayName,
        description: newWoundType.description,
        instructions: newWoundType.instructions,
        isEnabled: newWoundType.isEnabled,
        priority: newWoundType.priority
      });
    }
  };

  const handleToggleWoundType = (id: number, enabled: boolean) => {
    toggleWoundTypeMutation.mutate({ id, enabled });
  };

  const handleDeleteWoundType = (id: number) => {
    if (confirm("Are you sure you want to delete this wound type? This action cannot be undone.")) {
      deleteWoundTypeMutation.mutate(id);
    }
  };

  const hasChanges = () => {
    if (!agentData) return false;
    const data = agentData as any;
    return (
      systemPrompts !== (data.systemPrompts || "") ||
      carePlanStructure !== (data.carePlanStructure || "") ||
      specificWoundCare !== (data.specificWoundCare || "") ||
      questionsGuidelines !== (data.questionsGuidelines || "") ||
      productRecommendations !== (data.productRecommendations || "")
    );
  };

  // Admin access check
  if (!authLoading && (!isAuthenticated || user?.role !== 'admin')) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Settings className="mx-auto h-12 w-12 text-gray-400" />
          <h1 className="mt-4 text-xl font-semibold text-gray-900">Access Denied</h1>
          <p className="mt-2 text-gray-600">Only administrators can access system settings.</p>
          <Button 
            onClick={() => setLocation(isAuthenticated ? '/my-cases' : '/')}
            className="mt-4"
          >
            {isAuthenticated ? 'Back to My Cases' : 'Go Home'}
          </Button>
        </div>
      </div>
    );
  }

  if (isLoading || authLoading) {
    return (
      <div className="min-h-screen bg-bg-light flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-medical-blue mx-auto mb-3"></div>
          <p className="text-gray-600">Loading settings...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-bg-light">
      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-4">
            <Link href="/my-cases">
              <Button variant="outline" size="sm">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to My Cases
              </Button>
            </Link>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 flex items-center">
                <Settings className="mr-3 h-8 w-8 text-medical-blue" />
                AI Configuration Settings
              </h1>
              <p className="text-gray-600 mt-2">
                Configure how the AI analyzes wounds and generates care plans
              </p>
            </div>
          </div>
        </div>

        {/* Settings Content */}
        <Card className="shadow-lg border-0">
          <CardHeader className="bg-gradient-to-r from-medical-blue to-blue-600 text-white">
            <CardTitle className="text-xl font-semibold flex items-center">
              <Settings className="h-6 w-6 mr-3" />
              AI Instructions Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            <Tabs defaultValue="system" className="w-full">
              <TabsList className="grid w-full grid-cols-5 mb-6">
                <TabsTrigger value="system">System Prompts</TabsTrigger>
                <TabsTrigger value="structure">Care Plan Structure</TabsTrigger>
                <TabsTrigger value="wound">Wound Types</TabsTrigger>
                <TabsTrigger value="questions">Questions Guidelines</TabsTrigger>
                <TabsTrigger value="products">Product Recommendations</TabsTrigger>
              </TabsList>

              <TabsContent value="system" className="space-y-4">
                <div>
                  <h3 className="text-lg font-semibold mb-2">System Prompts</h3>
                  <p className="text-gray-600 mb-4">
                    Core mission and behavior instructions for the AI assistant.
                  </p>
                  <Textarea
                    value={systemPrompts}
                    onChange={(e) => setSystemPrompts(e.target.value)}
                    rows={20}
                    className="font-mono text-sm"
                    placeholder="Enter system prompts..."
                  />
                </div>
              </TabsContent>

              <TabsContent value="structure" className="space-y-4">
                <div>
                  <h3 className="text-lg font-semibold mb-2">Care Plan Structure</h3>
                  <p className="text-gray-600 mb-4">
                    How the AI should format and organize care plan responses.
                  </p>
                  <Textarea
                    value={carePlanStructure}
                    onChange={(e) => setCarePlanStructure(e.target.value)}
                    rows={20}
                    className="font-mono text-sm"
                    placeholder="Enter care plan structure instructions..."
                  />
                </div>
              </TabsContent>

              <TabsContent value="wound" className="space-y-4">
                <div>
                  <h3 className="text-lg font-semibold mb-2">Wound Type Management</h3>
                  <p className="text-gray-600 mb-4">
                    Configure AI instructions for different wound types. Select a wound type to edit its specific assessment guidelines.
                  </p>
                  
                  {/* Wound Type Selector */}
                  <div className="space-y-4">
                    <div>
                      <Label htmlFor="woundType">Select Wound Type</Label>
                      <Select value={selectedWoundType} onValueChange={handleWoundTypeChange}>
                        <SelectTrigger>
                          <SelectValue placeholder="Choose a wound type to edit..." />
                        </SelectTrigger>
                        <SelectContent>
                          {woundTypes && (woundTypes as any[]).map((type) => (
                            <SelectItem key={type.id} value={type.id.toString()}>
                              {type.displayName}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    
                    {selectedWoundType && (
                      <div className="space-y-4">
                        <div>
                          <Label htmlFor="woundInstructions">AI Instructions for This Wound Type</Label>
                          <Textarea
                            id="woundInstructions"
                            value={woundTypeInstructions}
                            onChange={(e) => handleWoundTypeInstructionsChange(e.target.value)}
                            rows={15}
                            className="font-mono text-sm"
                            placeholder="Enter specific AI instructions for this wound type..."
                          />
                        </div>
                        
                        <div className="flex justify-between">
                          <Button
                            onClick={handleSaveWoundType}
                            disabled={updateWoundTypeMutation.isPending}
                            className="bg-medical-blue hover:bg-medical-blue/90"
                          >
                            <Save className="h-4 w-4 mr-2" />
                            {updateWoundTypeMutation.isPending ? 'Saving...' : 'Save Instructions'}
                          </Button>
                        </div>
                      </div>
                    )}
                    
                    {/* Wound Types List */}
                    <div className="mt-8">
                      <div className="flex justify-between items-center mb-4">
                        <h4 className="text-md font-semibold">Manage Wound Types</h4>
                        <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
                          <DialogTrigger asChild>
                            <Button variant="outline" size="sm">
                              <Plus className="h-4 w-4 mr-2" />
                              Add New
                            </Button>
                          </DialogTrigger>
                          <DialogContent>
                            <DialogHeader>
                              <DialogTitle>Add New Wound Type</DialogTitle>
                            </DialogHeader>
                            <div className="space-y-4">
                              <div>
                                <Label htmlFor="newName">Name (Internal)</Label>
                                <Input
                                  id="newName"
                                  value={newWoundType.name}
                                  onChange={(e) => setNewWoundType({...newWoundType, name: e.target.value})}
                                  placeholder="e.g., burn_wound"
                                />
                              </div>
                              <div>
                                <Label htmlFor="newDisplayName">Display Name</Label>
                                <Input
                                  id="newDisplayName"
                                  value={newWoundType.displayName}
                                  onChange={(e) => setNewWoundType({...newWoundType, displayName: e.target.value})}
                                  placeholder="e.g., Burn Wound"
                                />
                              </div>
                              <div>
                                <Label htmlFor="newDescription">Description</Label>
                                <Input
                                  id="newDescription"
                                  value={newWoundType.description}
                                  onChange={(e) => setNewWoundType({...newWoundType, description: e.target.value})}
                                  placeholder="Brief description of this wound type"
                                />
                              </div>
                              <div>
                                <Label htmlFor="newInstructions">AI Instructions</Label>
                                <Textarea
                                  id="newInstructions"
                                  value={newWoundType.instructions}
                                  onChange={(e) => setNewWoundType({...newWoundType, instructions: e.target.value})}
                                  rows={8}
                                  placeholder="Enter AI instructions for this wound type..."
                                />
                              </div>
                              <div className="flex items-center space-x-2">
                                <Switch
                                  id="newEnabled"
                                  checked={newWoundType.isEnabled}
                                  onCheckedChange={(checked) => setNewWoundType({...newWoundType, isEnabled: checked})}
                                />
                                <Label htmlFor="newEnabled">Enable this wound type</Label>
                              </div>
                              <div className="flex justify-end space-x-2">
                                <Button variant="outline" onClick={() => setDialogOpen(false)}>
                                  Cancel
                                </Button>
                                <Button 
                                  onClick={handleCreateWoundType}
                                  disabled={createWoundTypeMutation.isPending}
                                >
                                  {createWoundTypeMutation.isPending ? 'Creating...' : 'Create'}
                                </Button>
                              </div>
                            </div>
                          </DialogContent>
                        </Dialog>
                      </div>
                      
                      <div className="grid gap-2">
                        {woundTypes && (woundTypes as any[]).map((type) => (
                          <div key={type.id} className="flex items-center justify-between p-3 border rounded-lg bg-gray-50">
                            <div className="flex items-center space-x-3">
                              <div className="flex items-center space-x-2">
                                <Switch
                                  checked={type.isEnabled}
                                  onCheckedChange={(checked) => handleToggleWoundType(type.id, checked)}
                                />
                                {type.isEnabled ? <Eye className="h-4 w-4 text-green-600" /> : <EyeOff className="h-4 w-4 text-gray-400" />}
                              </div>
                              <div>
                                <div className="font-medium">{type.displayName}</div>
                                <div className="text-sm text-gray-600">{type.description}</div>
                              </div>
                            </div>
                            <div className="flex items-center space-x-2">
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => setSelectedWoundType(type.id.toString())}
                              >
                                <Edit className="h-4 w-4" />
                              </Button>
                              {!type.isDefault && (
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => handleDeleteWoundType(type.id)}
                                  className="text-red-600 hover:text-red-700"
                                >
                                  <Trash2 className="h-4 w-4" />
                                </Button>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="questions" className="space-y-4">
                <div>
                  <h3 className="text-lg font-semibold mb-2">Questions Guidelines</h3>
                  <p className="text-gray-600 mb-4">
                    How to ask follow-up questions to improve diagnostic accuracy.
                  </p>
                  <Textarea
                    value={questionsGuidelines}
                    onChange={(e) => setQuestionsGuidelines(e.target.value)}
                    rows={20}
                    className="font-mono text-sm"
                    placeholder="Enter questions guidelines..."
                  />
                </div>
              </TabsContent>

              <TabsContent value="products" className="space-y-4">
                <div>
                  <h3 className="text-lg font-semibold mb-2">Product Recommendations</h3>
                  <p className="text-gray-600 mb-4">
                    Guidelines for recommending specific medical products and supplies. The system will generate dynamic Amazon search links for products.
                  </p>
                  
                  {/* Helpful Tips Card */}
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                    <h4 className="font-semibold text-blue-800 mb-2">Dynamic Product Links</h4>
                    <p className="text-blue-700 text-sm mb-2">
                      The system automatically generates working Amazon search links. You don't need to include specific URLs here.
                    </p>
                    <div className="text-blue-700 text-sm space-y-1">
                      <p><strong>Example Link Format:</strong> [Foam Dressing](https://www.amazon.com/s?k=foam+dressing+wounds)</p>
                      <p><strong>Focus on:</strong> Product categories, wound-specific recommendations, and usage guidelines</p>
                    </div>
                  </div>

                  {/* Example Content Card */}
                  <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 mb-4">
                    <h4 className="font-semibold text-gray-800 mb-2">Example Content Structure:</h4>
                    <pre className="text-xs text-gray-600 whitespace-pre-wrap">{`For Pressure Ulcers:
- Pressure redistribution: foam mattress overlays, cushions
- Wound dressings: hydrocolloid, foam, or alginate based on exudate level
- Skin protection: barrier creams, protective films

For Diabetic Ulcers:
- Offloading devices: specialized shoes, boot walkers
- Infection control: antimicrobial dressings, topical antibiotics
- Moisture management: absorbent dressings for moderate exudate`}</pre>
                  </div>

                  <Textarea
                    value={productRecommendations}
                    onChange={(e) => setProductRecommendations(e.target.value)}
                    rows={20}
                    className="font-mono text-sm"
                    placeholder="Enter product recommendation guidelines by wound type, severity, and care stage..."
                  />
                </div>
              </TabsContent>
            </Tabs>

            {/* Action Buttons */}
            <div className="flex justify-between items-center pt-6 border-t mt-6">
              <div className="text-sm text-gray-500">
                {/* Last updated timestamp would appear here */}
              </div>
              <div className="flex space-x-3">
                <Button
                  variant="outline"
                  onClick={handleReset}
                  disabled={updateMutation.isPending || !hasChanges()}
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Reset
                </Button>
                <Button
                  onClick={handleSave}
                  disabled={updateMutation.isPending || !hasChanges()}
                  className="bg-medical-blue hover:bg-blue-700"
                >
                  {updateMutation.isPending ? (
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Save className="h-4 w-4 mr-2" />
                  )}
                  Save Changes
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default SettingsPage;