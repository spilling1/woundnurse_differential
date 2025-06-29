import { Server } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { useQuery } from "@tanstack/react-query";

export default function SystemStatus() {
  const { data: status } = useQuery({
    queryKey: ['/api/status'],
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  return (
    <Card className="mt-8">
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Server className="text-medical-blue text-lg mr-2" />
            <h3 className="text-lg font-semibold text-gray-900">System Status</h3>
          </div>
          <div className="text-sm text-gray-500">
            Version {status?.version || 'v1.0.0'}
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
          <div className="bg-green-50 rounded-lg p-4">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-success-green rounded-full mr-2"></div>
              <span className="text-sm font-medium text-gray-900">API Status</span>
            </div>
            <p className="text-xs text-gray-600 mt-1">
              {status?.status === 'operational' ? 'All systems operational' : 'Checking status...'}
            </p>
          </div>
          
          <div className="bg-blue-50 rounded-lg p-4">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-medical-blue rounded-full mr-2"></div>
              <span className="text-sm font-medium text-gray-900">AI Models</span>
            </div>
            <p className="text-xs text-gray-600 mt-1">
              {status?.models?.length || 3} models available
            </p>
          </div>
          
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-neutral-gray rounded-full mr-2"></div>
              <span className="text-sm font-medium text-gray-900">Processing</span>
            </div>
            <p className="text-xs text-gray-600 mt-1">
              {status?.processingQueue || 0} in queue
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
