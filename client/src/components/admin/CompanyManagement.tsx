import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Building, Users, Plus, Edit, Trash2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface Company {
  id: number;
  name: string;
  domain: string;
  contactEmail: string;
  userCount: number;
  isActive: boolean;
}

interface CompanyManagementProps {
  companies: Company[];
  isLoading: boolean;
}

export function CompanyManagement({ companies, isLoading }: CompanyManagementProps) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [editingCompany, setEditingCompany] = useState<Company | null>(null);
  const [formData, setFormData] = useState({
    name: '',
    domain: '',
    contactEmail: ''
  });

  const createCompany = useMutation({
    mutationFn: async (data: typeof formData) => {
      const response = await fetch('/api/admin/companies', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
      if (!response.ok) throw new Error('Failed to create company');
      return response.json();
    },
    onSuccess: () => {
      toast({ title: "Success", description: "Company created successfully" });
      queryClient.invalidateQueries({ queryKey: ['/api/admin/companies'] });
      setShowCreateDialog(false);
      setFormData({ name: '', domain: '', contactEmail: '' });
    },
    onError: (error: any) => {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    }
  });

  const updateCompany = useMutation({
    mutationFn: async ({ id, data }: { id: number; data: typeof formData }) => {
      const response = await fetch(`/api/admin/companies/${id}`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
      if (!response.ok) throw new Error('Failed to update company');
      return response.json();
    },
    onSuccess: () => {
      toast({ title: "Success", description: "Company updated successfully" });
      queryClient.invalidateQueries({ queryKey: ['/api/admin/companies'] });
      setEditingCompany(null);
      setFormData({ name: '', domain: '', contactEmail: '' });
    },
    onError: (error: any) => {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    }
  });

  const deleteCompany = useMutation({
    mutationFn: async (id: number) => {
      const response = await fetch(`/api/admin/companies/${id}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
        },
      });
      if (!response.ok) throw new Error('Failed to delete company');
      return response.json();
    },
    onSuccess: () => {
      toast({ title: "Success", description: "Company deleted successfully" });
      queryClient.invalidateQueries({ queryKey: ['/api/admin/companies'] });
    },
    onError: (error: any) => {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    }
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (editingCompany) {
      updateCompany.mutate({ id: editingCompany.id, data: formData });
    } else {
      createCompany.mutate(formData);
    }
  };

  const handleEdit = (company: Company) => {
    setEditingCompany(company);
    setFormData({
      name: company.name,
      domain: company.domain,
      contactEmail: company.contactEmail
    });
  };

  const handleClose = () => {
    setEditingCompany(null);
    setShowCreateDialog(false);
    setFormData({ name: '', domain: '', contactEmail: '' });
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Building className="h-5 w-5" />
            Company Management
          </CardTitle>
          <CardDescription>Loading companies...</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Building className="h-5 w-5" />
          Company Management
        </CardTitle>
        <CardDescription>
          Manage organizational accounts and user groups
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <p className="text-sm text-gray-600">
              {companies.length} {companies.length === 1 ? 'company' : 'companies'} registered
            </p>
            <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
              <DialogTrigger asChild>
                <Button className="bg-medical-blue hover:bg-blue-700">
                  <Plus className="h-4 w-4 mr-2" />
                  Add Company
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Create New Company</DialogTitle>
                </DialogHeader>
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div>
                    <Label htmlFor="name">Company Name</Label>
                    <Input
                      id="name"
                      value={formData.name}
                      onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                      required
                    />
                  </div>
                  <div>
                    <Label htmlFor="domain">Domain</Label>
                    <Input
                      id="domain"
                      value={formData.domain}
                      onChange={(e) => setFormData({ ...formData, domain: e.target.value })}
                      placeholder="example.com"
                      required
                    />
                  </div>
                  <div>
                    <Label htmlFor="contactEmail">Contact Email</Label>
                    <Input
                      id="contactEmail"
                      type="email"
                      value={formData.contactEmail}
                      onChange={(e) => setFormData({ ...formData, contactEmail: e.target.value })}
                      required
                    />
                  </div>
                  <div className="flex justify-end gap-2">
                    <Button type="button" variant="outline" onClick={handleClose}>
                      Cancel
                    </Button>
                    <Button type="submit" disabled={createCompany.isPending}>
                      Create Company
                    </Button>
                  </div>
                </form>
              </DialogContent>
            </Dialog>
          </div>

          {companies.map((company) => (
            <div key={company.id} className="flex items-center justify-between p-4 border rounded-lg">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <h4 className="font-medium">{company.name}</h4>
                  <Badge variant={company.isActive ? 'default' : 'secondary'}>
                    {company.isActive ? 'Active' : 'Inactive'}
                  </Badge>
                </div>
                <p className="text-sm text-gray-600">{company.domain}</p>
                <p className="text-sm text-gray-500">{company.contactEmail}</p>
                <div className="flex items-center gap-1 text-xs text-gray-500">
                  <Users className="h-3 w-3" />
                  <span>{company.userCount} users</span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Dialog open={editingCompany?.id === company.id} onOpenChange={(open) => !open && handleClose()}>
                  <DialogTrigger asChild>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleEdit(company)}
                    >
                      <Edit className="h-4 w-4" />
                    </Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Edit Company</DialogTitle>
                    </DialogHeader>
                    <form onSubmit={handleSubmit} className="space-y-4">
                      <div>
                        <Label htmlFor="edit-name">Company Name</Label>
                        <Input
                          id="edit-name"
                          value={formData.name}
                          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                          required
                        />
                      </div>
                      <div>
                        <Label htmlFor="edit-domain">Domain</Label>
                        <Input
                          id="edit-domain"
                          value={formData.domain}
                          onChange={(e) => setFormData({ ...formData, domain: e.target.value })}
                          required
                        />
                      </div>
                      <div>
                        <Label htmlFor="edit-contactEmail">Contact Email</Label>
                        <Input
                          id="edit-contactEmail"
                          type="email"
                          value={formData.contactEmail}
                          onChange={(e) => setFormData({ ...formData, contactEmail: e.target.value })}
                          required
                        />
                      </div>
                      <div className="flex justify-end gap-2">
                        <Button type="button" variant="outline" onClick={handleClose}>
                          Cancel
                        </Button>
                        <Button type="submit" disabled={updateCompany.isPending}>
                          Update Company
                        </Button>
                      </div>
                    </form>
                  </DialogContent>
                </Dialog>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => deleteCompany.mutate(company.id)}
                  disabled={deleteCompany.isPending}
                  className="text-red-600 hover:text-red-700 hover:bg-red-50"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}