import { useQuery } from "@tanstack/react-query";

export function useAuth() {
  const { data: user, isLoading, error } = useQuery({
    queryKey: ["/api/auth/user"],
    retry: false,
    staleTime: Infinity, // Cache forever until explicit refresh
    gcTime: Infinity, // Don't garbage collect
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    refetchInterval: false,
    throwOnError: false, // Don't throw on 401 errors
    refetchOnMount: false, // Prevent refetching on every mount
    enabled: true, // Only fetch once
  });

  // Handle 401 errors gracefully - user is simply not authenticated
  const isAuthenticated = !!user && !error;

  return {
    user,
    isLoading,
    isAuthenticated,
  };
}