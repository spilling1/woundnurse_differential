import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiRequest } from '@/lib/queryClient';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Trash2, Edit, Plus, Eye, EyeOff } from 'lucide-react';
import { Link } from 'wouter';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { toast } from '@/hooks/use-toast';

interface ProductRecommendation {
  id: number;
  name: string;
  category: string;
  description: string;
  amazonSearchUrl: string;
  searchKeywords: string;
  woundTypes: string[];
  audiences: string[];
  priority: number;
  isActive: boolean;
  timesRecommended: number;
  extractedFromCaseId: string;
  createdAt: string;
  updatedAt: string;
}

export default function ProductManagement() {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [editingProduct, setEditingProduct] = useState<ProductRecommendation | null>(null);
  const queryClient = useQueryClient();

  const { data: products = [], isLoading, error } = useQuery<ProductRecommendation[]>({
    queryKey: ['/api/products/all'],
  });

  const toggleProductMutation = useMutation({
    mutationFn: async ({ id, isActive }: { id: number; isActive: boolean }) => {
      return apiRequest('PATCH', `/api/products/${id}/toggle`, { isActive });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/products/all'] });
      toast({ title: 'Product updated successfully' });
    },
    onError: () => {
      toast({ title: 'Failed to update product', variant: 'destructive' });
    }
  });

  const deleteProductMutation = useMutation({
    mutationFn: async (id: number) => {
      return apiRequest('DELETE', `/api/products/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/products/all'] });
      toast({ title: 'Product deleted successfully' });
    },
    onError: () => {
      toast({ title: 'Failed to delete product', variant: 'destructive' });
    }
  });

  if (isLoading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center min-h-96">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-2 text-gray-600">Loading products...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto p-6">
        <div className="text-center text-red-600">
          <p>Error loading products. Please try again.</p>
        </div>
      </div>
    );
  }

  // Get unique categories for filtering
  const categories = ['all', ...Array.from(new Set(products.map(p => p.category)))];
  
  // Filter products by category
  const filteredProducts = selectedCategory === 'all' 
    ? products 
    : products.filter(p => p.category === selectedCategory);

  // Group products by category for display
  const productsByCategory = filteredProducts.reduce((acc, product) => {
    if (!acc[product.category]) {
      acc[product.category] = [];
    }
    acc[product.category].push(product);
    return acc;
  }, {} as Record<string, ProductRecommendation[]>);

  const categoryDisplayNames = {
    'wound_dressing': 'Wound Dressings',
    'cleansing': 'Cleansing Products',
    'moisturizing': 'Moisturizing Products',
    'compression': 'Compression Products',
    'positioning': 'Positioning Products',
    'securing': 'Securing Products',
    'general': 'General Products',
    'medication': 'Medications'
  };

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <Link href="/admin">
            <Button variant="outline" size="sm">← Back to Admin</Button>
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Product Recommendations</h1>
            <p className="text-gray-600">Manage product database extracted from care plans</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Select value={selectedCategory} onValueChange={setSelectedCategory}>
            <SelectTrigger className="w-48">
              <SelectValue placeholder="Filter by category" />
            </SelectTrigger>
            <SelectContent>
              {categories.map(category => (
                <SelectItem key={category} value={category}>
                  {category === 'all' ? 'All Categories' : categoryDisplayNames[category] || category}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-blue-600">{products.length}</div>
            <div className="text-sm text-gray-600">Total Products</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-green-600">{products.filter(p => p.isActive).length}</div>
            <div className="text-sm text-gray-600">Active Products</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-purple-600">{Object.keys(categoryDisplayNames).length}</div>
            <div className="text-sm text-gray-600">Categories</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-orange-600">
              {products.reduce((sum, p) => sum + (p.timesRecommended || 0), 0)}
            </div>
            <div className="text-sm text-gray-600">Total Recommendations</div>
          </CardContent>
        </Card>
      </div>

      {/* Products by Category */}
      <div className="space-y-6">
        {Object.entries(productsByCategory).map(([category, categoryProducts]) => (
          <Card key={category}>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>{categoryDisplayNames[category] || category} ({categoryProducts.length})</span>
                <Badge variant="outline">
                  {categoryProducts.filter(p => p.isActive).length} active
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                {categoryProducts.map((product) => (
                  <div key={product.id} className={`border rounded-lg p-4 ${!product.isActive ? 'bg-gray-50 border-gray-200' : 'bg-white border-gray-300'}`}>
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <h3 className={`font-semibold ${!product.isActive ? 'text-gray-600' : 'text-gray-900'}`}>
                            {product.name}
                          </h3>
                          {!product.isActive && <Badge variant="secondary">Inactive</Badge>}
                          <Badge variant="outline">Priority: {product.priority}</Badge>
                          {product.timesRecommended > 0 && (
                            <Badge variant="outline" className="text-green-600">
                              Used {product.timesRecommended}x
                            </Badge>
                          )}
                        </div>
                        
                        {product.description && (
                          <p className={`text-sm mb-2 ${!product.isActive ? 'text-gray-500' : 'text-gray-700'}`}>
                            {product.description}
                          </p>
                        )}
                        
                        <div className="flex flex-wrap gap-2 text-xs">
                          {product.woundTypes?.map(type => (
                            <Badge key={type} variant="outline" className="text-blue-600">
                              {type.replace('_', ' ')}
                            </Badge>
                          ))}
                          {product.audiences?.map(audience => (
                            <Badge key={audience} variant="outline" className="text-purple-600">
                              {audience}
                            </Badge>
                          ))}
                        </div>

                        {product.searchKeywords && (
                          <p className="text-xs text-gray-500 mt-1">
                            Keywords: {product.searchKeywords}
                          </p>
                        )}

                        {product.extractedFromCaseId && (
                          <p className="text-xs text-gray-400 mt-1">
                            Extracted from case: {product.extractedFromCaseId}
                          </p>
                        )}
                      </div>
                      
                      <div className="flex items-center gap-2 ml-4">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => toggleProductMutation.mutate({ id: product.id, isActive: !product.isActive })}
                          disabled={toggleProductMutation.isPending}
                        >
                          {product.isActive ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => deleteProductMutation.mutate(product.id)}
                          disabled={deleteProductMutation.isPending}
                          className="text-red-600 hover:text-red-700"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                    
                    {product.amazonSearchUrl && (
                      <div className="mt-2 pt-2 border-t border-gray-200">
                        <a 
                          href={product.amazonSearchUrl} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-xs text-blue-600 hover:text-blue-800 underline"
                        >
                          View on Amazon →
                        </a>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {filteredProducts.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-500">No products found for the selected category.</p>
        </div>
      )}
    </div>
  );
}