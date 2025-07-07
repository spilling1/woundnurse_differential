import { useQuery, useQueryClient } from "@tanstack/react-query";

export function useAuth() {
  const queryClient = useQueryClient();
  
  // Check if we have a token in localStorage
  const token = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null;
  
  const { data: user, isLoading, error } = useQuery({
    queryKey: ["/api/auth/user"],
    queryFn: async () => {
      if (!token) {
        throw new Error('No token');
      }
      
      const response = await fetch('/api/auth/user', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      
      if (!response.ok) {
        if (response.status === 401) {
          // Token is invalid, remove it
          localStorage.removeItem('auth_token');
        }
        throw new Error('Authentication failed');
      }
      
      return response.json();
    },
    retry: false,
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
    gcTime: 10 * 60 * 1000, // Keep in cache for 10 minutes
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    refetchInterval: false,
    throwOnError: false, // Don't throw on 401 errors
    enabled: !!token, // Only fetch if we have a token
  });

  // Handle 401 errors gracefully - user is simply not authenticated
  const isAuthenticated = !!user && !error && !!token;

  // Logout function
  const logout = () => {
    localStorage.removeItem('auth_token');
    queryClient.clear(); // Clear all cached data
    window.location.href = '/login'; // Force redirect to login
  };

  return {
    user,
    isLoading,
    isAuthenticated,
    logout,
  };
}