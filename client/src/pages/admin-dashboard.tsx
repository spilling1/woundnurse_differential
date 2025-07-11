import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { 
  Users, 
  Building, 
  FileText, 
  Activity, 
  Settings, 
  LogOut,
  Eye,
  Edit3,
  Trash2,
  Plus,
  Shield,
  AlertCircle,
  CheckCircle,
  XCircle,
  Search,
  Filter,
  RotateCcw,
  Package,
  Grid,
  TrendingUp
} from "lucide-react";
import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useAuth } from "@/hooks/useAuth";
import type { User, Company, WoundAssessment, DetectionModel, AiAnalysisModel } from "@shared/schema";

interface DashboardStats {
  totalUsers: number;
  activeUsers: number;
  totalAssessments: number;
  totalCompanies: number;
  recentUsers: User[];
  recentAssessments: WoundAssessment[];
  usersByRole: {
    admin: number;
    user: number;
    nurse: number;
    manager: number;
  };
}

export default function AdminDashboard() {
  const [, setLocation] = useLocation();
  const { user: authUser, isAuthenticated } = useAuth();
  const user = authUser as User | undefined;
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("dashboard");
  const [userFilter, setUserFilter] = useState("");
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [selectedCompany, setSelectedCompany] = useState<Company | null>(null);
  const [selectedDetectionModel, setSelectedDetectionModel] = useState<DetectionModel | null>(null);

  // Redirect if not admin
  useEffect(() => {
    if (isAuthenticated && user && user.role !== 'admin') {
      setLocation('/');
      toast({
        title: "Access Denied",
        description: "You don't have permission to access the admin dashboard.",
        variant: "destructive",
      });
    }
  }, [isAuthenticated, user, setLocation, toast]);

  // Fetch dashboard data
  const { data: dashboardData, isLoading: isDashboardLoading } = useQuery<DashboardStats>({
    queryKey: ['/api/admin/dashboard'],
    enabled: isAuthenticated && user?.role === 'admin',
  });

  const { data: users = [], isLoading: isUsersLoading } = useQuery<User[]>({
    queryKey: ['/api/admin/users'],
    enabled: isAuthenticated && user?.role === 'admin',
  });

  const { data: companies = [], isLoading: isCompaniesLoading } = useQuery<Company[]>({
    queryKey: ['/api/admin/companies'],
    enabled: isAuthenticated && user?.role === 'admin',
  });

  const { data: assessments = [], isLoading: isAssessmentsLoading } = useQuery<WoundAssessment[]>({
    queryKey: ['/api/admin/assessments'],
    enabled: isAuthenticated && user?.role === 'admin',
  });

  const { data: detectionModels = [], isLoading: isDetectionModelsLoading } = useQuery<DetectionModel[]>({
    queryKey: ['/api/admin/detection-models'],
    enabled: isAuthenticated && user?.role === 'admin',
  });

  const { data: aiAnalysisModels = [], isLoading: isAiAnalysisModelsLoading } = useQuery<AiAnalysisModel[]>({
    queryKey: ['/api/admin/ai-analysis-models'],
    enabled: isAuthenticated && user?.role === 'admin',
  });

  // User management mutations
  const updateUserMutation = useMutation({
    mutationFn: async ({ userId, data }: { userId: string; data: any }) => {
      return apiRequest('PUT', `/api/admin/users/${userId}`, data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/admin/users'] });
      queryClient.invalidateQueries({ queryKey: ['/api/admin/dashboard'] });
      toast({
        title: "Success",
        description: "User updated successfully",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Error",
        description: error.message || "Failed to update user",
        variant: "destructive",
      });
    },
  });

  const deleteUserMutation = useMutation({
    mutationFn: async (userId: string) => {
      return apiRequest('DELETE', `/api/admin/users/${userId}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/admin/users'] });
      queryClient.invalidateQueries({ queryKey: ['/api/admin/dashboard'] });
      toast({
        title: "Success",
        description: "User deleted successfully",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Error",
        description: error.message || "Failed to delete user",
        variant: "destructive",
      });
    },
  });

  const toggleUserRoleMutation = useMutation({
    mutationFn: async ({ userId, newRole }: { userId: string; newRole: string }) => {
      return apiRequest('PUT', `/api/admin/users/${userId}`, { role: newRole });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/admin/users'] });
      queryClient.invalidateQueries({ queryKey: ['/api/admin/dashboard'] });
      toast({
        title: "Success",
        description: "User role updated successfully",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Error",
        description: error.message || "Failed to update user role",
        variant: "destructive",
      });
    },
  });

  // Company management mutations
  const createCompanyMutation = useMutation({
    mutationFn: async (data: any) => {
      return apiRequest('POST', '/api/admin/companies', data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/admin/companies'] });
      queryClient.invalidateQueries({ queryKey: ['/api/admin/dashboard'] });
      toast({
        title: "Success",
        description: "Company created successfully",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Error",
        description: error.message || "Failed to create company",
        variant: "destructive",
      });
    },
  });

  const updateCompanyMutation = useMutation({
    mutationFn: async ({ id, data }: { id: number; data: any }) => {
      return apiRequest('PUT', `/api/admin/companies/${id}`, data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/admin/companies'] });
      queryClient.invalidateQueries({ queryKey: ['/api/admin/dashboard'] });
      toast({
        title: "Success",
        description: "Company updated successfully",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Error",
        description: error.message || "Failed to update company",
        variant: "destructive",
      });
    },
  });

  const deleteCompanyMutation = useMutation({
    mutationFn: async (id: number) => {
      return apiRequest('DELETE', `/api/admin/companies/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/admin/companies'] });
      queryClient.invalidateQueries({ queryKey: ['/api/admin/dashboard'] });
      toast({
        title: "Success",
        description: "Company deleted successfully",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Error",
        description: error.message || "Failed to delete company",
        variant: "destructive",
      });
    },
  });

  // Detection model management mutations
  const toggleDetectionModelMutation = useMutation({
    mutationFn: async ({ id, enabled }: { id: number; enabled: boolean }) => {
      return apiRequest('PATCH', `/api/admin/detection-models/${id}/toggle`, { enabled });
    },
    onMutate: async ({ id, enabled }) => {
      // Cancel any outgoing refetches so they don't overwrite our optimistic update
      await queryClient.cancelQueries({ queryKey: ['/api/admin/detection-models'] });

      // Snapshot the previous value
      const previousModels = queryClient.getQueryData(['/api/admin/detection-models']);

      // Optimistically update to the new value
      queryClient.setQueryData(['/api/admin/detection-models'], (old: any) => {
        if (!old) return old;
        return old.map((model: any) => 
          model.id === id ? { ...model, isEnabled: enabled } : model
        );
      });

      // Return a context object with the snapshotted value
      return { previousModels };
    },
    onError: (err, variables, context) => {
      // If the mutation fails, use the context returned from onMutate to roll back
      if (context?.previousModels) {
        queryClient.setQueryData(['/api/admin/detection-models'], context.previousModels);
      }
      toast({
        title: "Error",
        description: "Failed to update detection model",
        variant: "destructive",
      });
    },
    onSuccess: () => {
      toast({
        title: "Success",
        description: "Detection model status updated successfully",
      });
    },
    onSettled: () => {
      // Always refetch after error or success to ensure we have latest data
      queryClient.invalidateQueries({ queryKey: ['/api/admin/detection-models'] });
    },
  });

  const updateDetectionModelMutation = useMutation({
    mutationFn: async ({ id, data }: { id: number; data: any }) => {
      return apiRequest('PUT', `/api/admin/detection-models/${id}`, data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/admin/detection-models'] });
      toast({
        title: "Success",
        description: "Detection model updated successfully",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Error",
        description: error.message || "Failed to update detection model",
        variant: "destructive",
      });
    },
  });

  const deleteDetectionModelMutation = useMutation({
    mutationFn: async (id: number) => {
      return apiRequest('DELETE', `/api/admin/detection-models/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/admin/detection-models'] });
      toast({
        title: "Success",
        description: "Detection model deleted successfully",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Error",
        description: error.message || "Failed to delete detection model",
        variant: "destructive",
      });
    },
  });

  // AI analysis model management mutations
  const toggleAiAnalysisModelMutation = useMutation({
    mutationFn: async ({ id, enabled }: { id: number; enabled: boolean }) => {
      return apiRequest('PATCH', `/api/admin/ai-analysis-models/${id}/toggle`, { enabled });
    },
    onMutate: async ({ id, enabled }) => {
      await queryClient.cancelQueries({ queryKey: ['/api/admin/ai-analysis-models'] });
      const previousModels = queryClient.getQueryData(['/api/admin/ai-analysis-models']);
      queryClient.setQueryData(['/api/admin/ai-analysis-models'], (old: any) => {
        if (!old) return old;
        return old.map((model: any) => 
          model.id === id ? { ...model, isEnabled: enabled } : model
        );
      });
      return { previousModels };
    },
    onError: (err, variables, context) => {
      if (context?.previousModels) {
        queryClient.setQueryData(['/api/admin/ai-analysis-models'], context.previousModels);
      }
      toast({
        title: "Error",
        description: "Failed to update AI analysis model",
        variant: "destructive",
      });
    },
    onSuccess: () => {
      toast({
        title: "Success",
        description: "AI analysis model status updated successfully",
      });
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/admin/ai-analysis-models'] });
      queryClient.invalidateQueries({ queryKey: ['/api/ai-analysis-models'] });
    },
  });

  const setDefaultAiAnalysisModelMutation = useMutation({
    mutationFn: async (id: number) => {
      return apiRequest('PATCH', `/api/admin/ai-analysis-models/${id}/default`, {});
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/admin/ai-analysis-models'] });
      queryClient.invalidateQueries({ queryKey: ['/api/ai-analysis-models'] });
      toast({
        title: "Success",
        description: "Default AI analysis model updated successfully",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Error",
        description: error.message || "Failed to set default AI analysis model",
        variant: "destructive",
      });
    },
  });

  // Filter users
  const filteredUsers = users.filter(user => 
    user.email?.toLowerCase().includes(userFilter.toLowerCase()) ||
    user.firstName?.toLowerCase().includes(userFilter.toLowerCase()) ||
    user.lastName?.toLowerCase().includes(userFilter.toLowerCase())
  );

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'admin': return 'bg-red-100 text-red-800';
      case 'nurse': return 'bg-blue-100 text-blue-800';
      case 'manager': return 'bg-purple-100 text-purple-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800';
      case 'inactive': return 'bg-yellow-100 text-yellow-800';
      case 'suspended': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (!isAuthenticated || user?.role !== 'admin') {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Shield className="mx-auto h-12 w-12 text-gray-400" />
          <h1 className="mt-4 text-xl font-semibold text-gray-900">Access Denied</h1>
          <p className="mt-2 text-gray-600">You don't have permission to access this page.</p>
          <Button 
            onClick={() => setLocation('/')}
            className="mt-4"
          >
            Go Home
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <Shield className="text-red-600 text-2xl mr-3" />
              <h1 className="text-xl font-semibold text-gray-900">Admin Dashboard</h1>
            </div>
            <div className="flex items-center space-x-4">
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => setLocation("/my-cases")}
                className="flex items-center"
                title="Back to My Cases"
              >
                <FileText className="mr-2 h-4 w-4" />
                Back to My Cases
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
                onClick={() => {
                  localStorage.removeItem('auth_token');
                  setLocation('/');
                }}
              >
                <LogOut className="mr-2 h-4 w-4" />
                Log Out
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-7">
            <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
            <TabsTrigger value="users">Users</TabsTrigger>
            <TabsTrigger value="companies">Companies</TabsTrigger>
            <TabsTrigger value="assessments">Assessments</TabsTrigger>
            <TabsTrigger value="detection-models">Detection Models</TabsTrigger>
            <TabsTrigger value="ai-analysis-models">AI Analysis Models</TabsTrigger>
            <TabsTrigger value="products">Products</TabsTrigger>
          </TabsList>

          {/* Dashboard Overview */}
          <TabsContent value="dashboard" className="space-y-6">
            {isDashboardLoading ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {Array.from({ length: 4 }).map((_, i) => (
                  <Card key={i} className="animate-pulse">
                    <CardContent className="pt-6">
                      <div className="h-8 bg-gray-200 rounded mb-2"></div>
                      <div className="h-6 bg-gray-200 rounded w-3/4"></div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : dashboardData ? (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Total Users</CardTitle>
                      <Users className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">{dashboardData.totalUsers}</div>
                      <p className="text-xs text-muted-foreground">
                        {dashboardData.activeUsers} active
                      </p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Total Companies</CardTitle>
                      <Building className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">{dashboardData.totalCompanies}</div>
                      <p className="text-xs text-muted-foreground">
                        Organizations
                      </p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Total Assessments</CardTitle>
                      <FileText className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">{dashboardData.totalAssessments}</div>
                      <p className="text-xs text-muted-foreground">
                        Wound assessments
                      </p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">System Health</CardTitle>
                      <Activity className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-green-600">Online</div>
                      <p className="text-xs text-muted-foreground">
                        All systems operational
                      </p>
                    </CardContent>
                  </Card>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card>
                    <CardHeader>
                      <CardTitle>Recent Users</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {dashboardData.recentUsers.map((user) => (
                          <div key={user.id} className="flex items-center justify-between">
                            <div className="flex items-center space-x-3">
                              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                                <span className="text-sm font-medium text-blue-700">
                                  {user.firstName?.[0] || user.email?.[0]?.toUpperCase() || '?'}
                                </span>
                              </div>
                              <div>
                                <p className="text-sm font-medium">{user.firstName} {user.lastName}</p>
                                <p className="text-xs text-gray-500">{user.email}</p>
                              </div>
                            </div>
                            <Badge className={getRoleColor(user.role)}>
                              {user.role}
                            </Badge>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>Users by Role</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {Object.entries(dashboardData.usersByRole).map(([role, count]) => (
                          <div key={role} className="flex items-center justify-between">
                            <span className="text-sm font-medium capitalize">{role}</span>
                            <Badge variant="outline">{count}</Badge>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </>
            ) : null}
          </TabsContent>

          {/* Users Management */}
          <TabsContent value="users" className="space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-bold">User Management</h2>
              <div className="flex space-x-2">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <Input 
                    placeholder="Search users..."
                    value={userFilter}
                    onChange={(e) => setUserFilter(e.target.value)}
                    className="pl-10 w-64"
                  />
                </div>
              </div>
            </div>

            {isUsersLoading ? (
              <div className="space-y-4">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Card key={i} className="animate-pulse">
                    <CardContent className="pt-6">
                      <div className="h-6 bg-gray-200 rounded mb-2"></div>
                      <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <div className="space-y-4">
                {filteredUsers.map((user) => (
                  <Card key={user.id}>
                    <CardContent className="pt-6">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                          <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                            <span className="text-sm font-medium text-blue-700">
                              {user.firstName?.[0] || user.email?.[0]?.toUpperCase() || '?'}
                            </span>
                          </div>
                          <div>
                            <h3 className="font-semibold">{user.firstName} {user.lastName}</h3>
                            <p className="text-sm text-gray-600">{user.email}</p>
                            <p className="text-xs text-gray-500">
                              Joined {new Date(user.createdAt!).toLocaleDateString()}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Badge className={getRoleColor(user.role)}>
                            {user.role}
                          </Badge>
                          <Badge className={getStatusColor(user.status)}>
                            {user.status}
                          </Badge>
                          <Button 
                            variant="ghost" 
                            size="sm"
                            onClick={() => {
                              const newRole = user.role === 'admin' ? 'user' : 'admin';
                              toggleUserRoleMutation.mutate({ userId: user.id, newRole });
                            }}
                            disabled={toggleUserRoleMutation.isPending}
                            title={user.role === 'admin' ? 'Remove Admin Role' : 'Make Admin'}
                          >
                            <RotateCcw className={`h-4 w-4 ${user.role === 'admin' ? 'text-red-500' : 'text-green-500'}`} />
                          </Button>
                          <Button 
                            variant="ghost" 
                            size="sm"
                            onClick={() => setSelectedUser(user)}
                          >
                            <Edit3 className="h-4 w-4" />
                          </Button>
                          <Button 
                            variant="ghost" 
                            size="sm"
                            onClick={() => deleteUserMutation.mutate(user.id)}
                            disabled={deleteUserMutation.isPending}
                          >
                            <Trash2 className="h-4 w-4 text-red-500" />
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>

          {/* Companies Management */}
          <TabsContent value="companies" className="space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-bold">Company Management</h2>
              <Button onClick={() => setSelectedCompany({} as Company)}>
                <Plus className="mr-2 h-4 w-4" />
                Add Company
              </Button>
            </div>

            {isCompaniesLoading ? (
              <div className="space-y-4">
                {Array.from({ length: 3 }).map((_, i) => (
                  <Card key={i} className="animate-pulse">
                    <CardContent className="pt-6">
                      <div className="h-6 bg-gray-200 rounded mb-2"></div>
                      <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <div className="space-y-4">
                {companies.map((company) => (
                  <Card key={company.id}>
                    <CardContent className="pt-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="font-semibold">{company.name}</h3>
                          <p className="text-sm text-gray-600">{company.domain}</p>
                          <p className="text-xs text-gray-500">
                            Max Users: {company.maxUsers} | Plan: {company.subscriptionPlan}
                          </p>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Badge className={getStatusColor(company.status)}>
                            {company.status}
                          </Badge>
                          <Button 
                            variant="ghost" 
                            size="sm"
                            onClick={() => setSelectedCompany(company)}
                          >
                            <Edit3 className="h-4 w-4" />
                          </Button>
                          <Button 
                            variant="ghost" 
                            size="sm"
                            onClick={() => deleteCompanyMutation.mutate(company.id)}
                            disabled={deleteCompanyMutation.isPending}
                          >
                            <Trash2 className="h-4 w-4 text-red-500" />
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>

          {/* Assessments View */}
          <TabsContent value="assessments" className="space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-bold">All Assessments</h2>
              <div className="text-sm text-gray-600">
                Total: {assessments.length}
              </div>
            </div>

            {isAssessmentsLoading ? (
              <div className="space-y-4">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Card key={i} className="animate-pulse">
                    <CardContent className="pt-6">
                      <div className="h-6 bg-gray-200 rounded mb-2"></div>
                      <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <div className="space-y-4">
                {assessments.map((assessment) => (
                  <Card key={assessment.id}>
                    <CardContent className="pt-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="font-semibold">Case {assessment.caseId}</h3>
                          <p className="text-sm text-gray-600">
                            User: {assessment.userId || 'Anonymous'} | 
                            Audience: {assessment.audience} | 
                            Model: {assessment.model}
                          </p>
                          <p className="text-xs text-gray-500">
                            {new Date(assessment.createdAt!).toLocaleDateString()} at {new Date(assessment.createdAt!).toLocaleTimeString()}
                          </p>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Badge variant="outline">
                            v{assessment.versionNumber}
                          </Badge>
                          {assessment.isFollowUp && (
                            <Badge className="bg-blue-100 text-blue-800">
                              Follow-up
                            </Badge>
                          )}
                          <Button 
                            variant="ghost" 
                            size="sm"
                            onClick={() => setLocation(`/care-plan/${assessment.caseId}`)}
                          >
                            <Eye className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>

          {/* Detection Models Tab */}
          <TabsContent value="detection-models" className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold">Detection Models</h3>
                <p className="text-sm text-gray-600">
                  Manage AI detection models used for wound analysis
                </p>
              </div>
              <div className="flex items-center space-x-2">
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => queryClient.invalidateQueries({ queryKey: ['/api/admin/detection-models'] })}
                >
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Refresh
                </Button>
              </div>
            </div>

            {isDetectionModelsLoading ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Array.from({ length: 4 }).map((_, i) => (
                  <Card key={i} className="animate-pulse">
                    <CardContent className="pt-6">
                      <div className="h-6 bg-gray-200 rounded mb-2"></div>
                      <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {detectionModels.map((model) => (
                  <Card key={model.id} className="relative">
                    <CardContent className="pt-6">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-2">
                            <h4 className="font-medium">{model.displayName}</h4>
                            <Badge 
                              variant={model.isEnabled ? "default" : "secondary"}
                              className={model.isEnabled ? "bg-green-100 text-green-800" : "bg-gray-100 text-gray-800"}
                            >
                              {model.isEnabled ? 'Enabled' : 'Disabled'}
                            </Badge>
                          </div>
                          <p className="text-sm text-gray-600 mb-3">
                            {model.description}
                          </p>
                          
                          {/* Capabilities and Accuracy */}
                          {model.capabilities && (
                            <div className="space-y-3 mb-3">
                              <div className="flex items-center justify-between">
                                <div className="flex items-center space-x-4 text-sm">
                                  <span className="font-medium text-gray-700">
                                    Status: {model.capabilities.overall_accuracy || model.capabilities.accuracy || 'Not evaluated'}
                                  </span>
                                  <span className="text-gray-500">•</span>
                                  <span className="text-gray-600">
                                    Speed: {model.capabilities.speed || 'Unknown'}
                                  </span>
                                </div>
                              </div>
                              
                              <div className="flex items-center space-x-3">
                                {model.capabilities.detection && (
                                  <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                                    Detection
                                  </Badge>
                                )}
                                {model.capabilities.measurements && (
                                  <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                                    Measurements
                                  </Badge>
                                )}
                                {model.capabilities.classification && (
                                  <Badge variant="outline" className="bg-purple-50 text-purple-700 border-purple-200">
                                    Classification
                                  </Badge>
                                )}
                              </div>
                              
                              {model.capabilities.capabilities_list && model.capabilities.capabilities_list.length > 0 && (
                                <div className="mt-2">
                                  <p className="text-xs font-medium text-gray-700 mb-1">Capabilities:</p>
                                  <div className="grid grid-cols-1 gap-1">
                                    {model.capabilities.capabilities_list.map((capability: string, index: number) => (
                                      <div key={index} className="text-xs text-gray-600 flex items-center">
                                        <span className="w-1 h-1 bg-gray-400 rounded-full mr-2"></span>
                                        {capability}
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          )}
                          
                          <div className="flex items-center space-x-2 text-sm text-gray-500 border-t pt-3">
                            <span>Priority: {model.priority}</span>
                            <span>•</span>
                            <span>Type: {model.name}</span>
                          </div>
                        </div>
                        <div className="flex items-center space-x-3">
                          <div className="flex items-center space-x-2">
                            <span className="text-sm font-medium">
                              {model.isEnabled ? 'ON' : 'OFF'}
                            </span>
                            <Switch
                              checked={model.isEnabled}
                              onCheckedChange={(checked) => 
                                toggleDetectionModelMutation.mutate({ 
                                  id: model.id, 
                                  enabled: checked 
                                })
                              }
                              disabled={toggleDetectionModelMutation.isPending}
                            />
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setSelectedDetectionModel(model)}
                          >
                            <Edit3 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>

          {/* AI Analysis Models Management */}
          <TabsContent value="ai-analysis-models" className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-medium">AI Analysis Models</h3>
                <p className="text-sm text-muted-foreground">
                  Manage AI models used for wound analysis and care plan generation
                </p>
              </div>
            </div>

            {isAiAnalysisModelsLoading ? (
              <div className="grid grid-cols-1 gap-4">
                {Array.from({ length: 3 }).map((_, i) => (
                  <Card key={i} className="animate-pulse">
                    <CardContent className="pt-6">
                      <div className="h-8 bg-gray-200 rounded mb-2"></div>
                      <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-4">
                {aiAnalysisModels.map((model) => (
                  <Card key={model.id} className="overflow-hidden">
                    <CardContent className="pt-6">
                      <div className="flex items-start justify-between">
                        <div className="flex-1 space-y-3">
                          <div className="flex items-center space-x-3">
                            <div className="flex items-center space-x-2">
                              <h4 className="font-medium">{model.displayName}</h4>
                              {model.isDefault && (
                                <Badge variant="default" className="text-xs">
                                  Default
                                </Badge>
                              )}
                              <Badge 
                                variant={model.isEnabled ? "default" : "secondary"} 
                                className="text-xs"
                              >
                                {model.isEnabled ? "Enabled" : "Disabled"}
                              </Badge>
                              <Badge variant="outline" className="text-xs">
                                {model.provider}
                              </Badge>
                              <Badge variant="outline" className="text-xs">
                                Priority: {model.priority}
                              </Badge>
                            </div>
                          </div>
                          
                          <p className="text-sm text-muted-foreground">
                            {model.description}
                          </p>
                          
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                              <span className="font-medium">Model ID:</span>
                              <span className="ml-2 text-muted-foreground">{model.modelId}</span>
                            </div>
                            <div>
                              <span className="font-medium">API Key:</span>
                              <span className="ml-2 text-muted-foreground">
                                {model.requiresApiKey ? model.apiKeyName : "Not required"}
                              </span>
                            </div>
                          </div>
                          
                          {model.capabilities && model.capabilities.length > 0 && (
                            <div>
                              <span className="text-sm font-medium">Capabilities:</span>
                              <div className="flex flex-wrap gap-1 mt-1">
                                {model.capabilities.map((capability) => (
                                  <Badge key={capability} variant="outline" className="text-xs">
                                    {capability}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                        
                        <div className="flex items-center space-x-2 ml-4">
                          {!model.isDefault && model.isEnabled && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => {
                                setDefaultAiAnalysisModelMutation.mutate(model.id);
                              }}
                              disabled={setDefaultAiAnalysisModelMutation.isPending}
                            >
                              Set Default
                            </Button>
                          )}
                          
                          <div className="flex items-center space-x-2">
                            <label htmlFor={`ai-model-${model.id}`} className="text-sm font-medium">
                              Enable
                            </label>
                            <Switch
                              id={`ai-model-${model.id}`}
                              checked={model.isEnabled}
                              onCheckedChange={(enabled) => {
                                toggleAiAnalysisModelMutation.mutate({ 
                                  id: model.id, 
                                  enabled 
                                });
                              }}
                              disabled={toggleAiAnalysisModelMutation.isPending}
                            />
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>
          {/* Products Management */}
          <TabsContent value="products" className="space-y-6">
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-2xl font-bold">Product Recommendations</h2>
                <p className="text-gray-600">Manage product database extracted from care plans</p>
              </div>
              <Link href="/admin/products">
                <Button>
                  <Eye className="mr-2 h-4 w-4" />
                  Manage Products
                </Button>
              </Link>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center space-x-2">
                    <Package className="h-8 w-8 text-blue-600" />
                    <div>
                      <p className="text-2xl font-bold">46</p>
                      <p className="text-sm text-gray-600">Total Products</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-8 w-8 text-green-600" />
                    <div>
                      <p className="text-2xl font-bold">46</p>
                      <p className="text-sm text-gray-600">Active Products</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center space-x-2">
                    <Grid className="h-8 w-8 text-purple-600" />
                    <div>
                      <p className="text-2xl font-bold">7</p>
                      <p className="text-sm text-gray-600">Categories</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center space-x-2">
                    <TrendingUp className="h-8 w-8 text-orange-600" />
                    <div>
                      <p className="text-2xl font-bold">-</p>
                      <p className="text-sm text-gray-600">Usage Data</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Product Categories</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-4 bg-blue-50 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">18</div>
                    <div className="text-sm text-gray-600">Wound Dressings</div>
                  </div>
                  <div className="text-center p-4 bg-green-50 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">11</div>
                    <div className="text-sm text-gray-600">General Products</div>
                  </div>
                  <div className="text-center p-4 bg-purple-50 rounded-lg">
                    <div className="text-2xl font-bold text-purple-600">7</div>
                    <div className="text-sm text-gray-600">Moisturizing</div>
                  </div>
                  <div className="text-center p-4 bg-orange-50 rounded-lg">
                    <div className="text-2xl font-bold text-orange-600">4</div>
                    <div className="text-sm text-gray-600">Cleansing</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* User Edit Modal */}
      {selectedUser && (
        <Dialog open={!!selectedUser} onOpenChange={() => setSelectedUser(null)}>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>Edit User</DialogTitle>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Role</label>
                <Select 
                  value={selectedUser.role} 
                  onValueChange={(value) => setSelectedUser({...selectedUser, role: value as any})}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="user">User</SelectItem>
                    <SelectItem value="nurse">Nurse</SelectItem>
                    <SelectItem value="manager">Manager</SelectItem>
                    <SelectItem value="admin">Admin</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Status</label>
                <Select 
                  value={selectedUser.status} 
                  onValueChange={(value) => setSelectedUser({...selectedUser, status: value as any})}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="active">Active</SelectItem>
                    <SelectItem value="inactive">Inactive</SelectItem>
                    <SelectItem value="suspended">Suspended</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex justify-end space-x-2">
                <Button variant="outline" onClick={() => setSelectedUser(null)}>
                  Cancel
                </Button>
                <Button 
                  onClick={() => {
                    updateUserMutation.mutate({
                      userId: selectedUser.id,
                      data: {
                        role: selectedUser.role,
                        status: selectedUser.status,
                      }
                    });
                    setSelectedUser(null);
                  }}
                  disabled={updateUserMutation.isPending}
                >
                  Save Changes
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      )}

      {/* Company Edit Modal */}
      {selectedCompany && (
        <Dialog open={!!selectedCompany} onOpenChange={() => setSelectedCompany(null)}>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>
                {selectedCompany.id ? 'Edit Company' : 'Add Company'}
              </DialogTitle>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Company Name</label>
                <Input 
                  value={selectedCompany.name || ''} 
                  onChange={(e) => setSelectedCompany({...selectedCompany, name: e.target.value})}
                  placeholder="Enter company name"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Domain</label>
                <Input 
                  value={selectedCompany.domain || ''} 
                  onChange={(e) => setSelectedCompany({...selectedCompany, domain: e.target.value})}
                  placeholder="company.com"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Max Users</label>
                <Input 
                  type="number"
                  value={selectedCompany.maxUsers || 100} 
                  onChange={(e) => setSelectedCompany({...selectedCompany, maxUsers: parseInt(e.target.value)})}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Subscription Plan</label>
                <Select 
                  value={selectedCompany.subscriptionPlan || 'basic'} 
                  onValueChange={(value) => setSelectedCompany({...selectedCompany, subscriptionPlan: value as any})}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="basic">Basic</SelectItem>
                    <SelectItem value="pro">Pro</SelectItem>
                    <SelectItem value="enterprise">Enterprise</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              {selectedCompany.id && (
                <div className="space-y-2">
                  <label className="text-sm font-medium">Status</label>
                  <Select 
                    value={selectedCompany.status || 'active'} 
                    onValueChange={(value) => setSelectedCompany({...selectedCompany, status: value as any})}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="active">Active</SelectItem>
                      <SelectItem value="inactive">Inactive</SelectItem>
                      <SelectItem value="suspended">Suspended</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              )}
              <div className="flex justify-end space-x-2">
                <Button variant="outline" onClick={() => setSelectedCompany(null)}>
                  Cancel
                </Button>
                <Button 
                  onClick={() => {
                    if (selectedCompany.id) {
                      updateCompanyMutation.mutate({
                        id: selectedCompany.id,
                        data: {
                          name: selectedCompany.name,
                          domain: selectedCompany.domain,
                          maxUsers: selectedCompany.maxUsers,
                          subscriptionPlan: selectedCompany.subscriptionPlan,
                          status: selectedCompany.status,
                        }
                      });
                    } else {
                      createCompanyMutation.mutate({
                        name: selectedCompany.name,
                        domain: selectedCompany.domain,
                        maxUsers: selectedCompany.maxUsers,
                        subscriptionPlan: selectedCompany.subscriptionPlan,
                      });
                    }
                    setSelectedCompany(null);
                  }}
                  disabled={updateCompanyMutation.isPending || createCompanyMutation.isPending}
                >
                  {selectedCompany.id ? 'Save Changes' : 'Create Company'}
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      )}

      {/* Detection Model Edit Modal */}
      {selectedDetectionModel && (
        <Dialog open={!!selectedDetectionModel} onOpenChange={() => setSelectedDetectionModel(null)}>
          <DialogContent className="sm:max-w-[500px]">
            <DialogHeader>
              <DialogTitle>Edit Detection Model</DialogTitle>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Display Name</label>
                <Input 
                  value={selectedDetectionModel.displayName || ''} 
                  onChange={(e) => setSelectedDetectionModel({
                    ...selectedDetectionModel, 
                    displayName: e.target.value
                  })}
                  placeholder="Enter display name"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Description</label>
                <Input 
                  value={selectedDetectionModel.description || ''} 
                  onChange={(e) => setSelectedDetectionModel({
                    ...selectedDetectionModel, 
                    description: e.target.value
                  })}
                  placeholder="Enter description"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Priority</label>
                <Input 
                  type="number"
                  value={selectedDetectionModel.priority || 1} 
                  onChange={(e) => setSelectedDetectionModel({
                    ...selectedDetectionModel, 
                    priority: parseInt(e.target.value)
                  })}
                  min="1"
                  max="10"
                />
                <p className="text-xs text-gray-500">
                  Lower numbers have higher priority (1 = highest priority)
                </p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Configuration (JSON)</label>
                <textarea 
                  className="w-full p-2 border rounded-md text-sm font-mono"
                  rows={6}
                  value={selectedDetectionModel.config ? JSON.stringify(selectedDetectionModel.config, null, 2) : '{}'}
                  onChange={(e) => {
                    try {
                      const config = JSON.parse(e.target.value);
                      setSelectedDetectionModel({
                        ...selectedDetectionModel, 
                        config: config
                      });
                    } catch (error) {
                      // Keep current value if JSON is invalid
                    }
                  }}
                  placeholder="Enter configuration as JSON"
                />
                <p className="text-xs text-gray-500">
                  Model-specific configuration parameters
                </p>
              </div>
              <div className="flex justify-end space-x-2">
                <Button variant="outline" onClick={() => setSelectedDetectionModel(null)}>
                  Cancel
                </Button>
                <Button 
                  onClick={() => {
                    updateDetectionModelMutation.mutate({
                      id: selectedDetectionModel.id,
                      data: {
                        displayName: selectedDetectionModel.displayName,
                        description: selectedDetectionModel.description,
                        priority: selectedDetectionModel.priority,
                        config: selectedDetectionModel.config,
                      }
                    });
                    setSelectedDetectionModel(null);
                  }}
                  disabled={updateDetectionModelMutation.isPending}
                >
                  Save Changes
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
}