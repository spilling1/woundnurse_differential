import { useQuery } from "@tanstack/react-query";
import { Circle, Eye, EyeOff } from "lucide-react";

interface DetectionStatus {
  status: string;
  method: string;
  service: string;
  version: string;
}

export default function ImageDetectionStatus() {
  const { data: detectionStatus, isLoading } = useQuery<DetectionStatus>({
    queryKey: ['/api/detection-status'],
    refetchInterval: 10000, // Check every 10 seconds
    retry: 1,
    staleTime: 5000,
  });

  if (isLoading) {
    return (
      <div className="text-sm text-gray-500">
        <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full text-xs font-medium">
          <Circle className="inline w-2 h-2 mr-1 fill-current animate-pulse" />
          Detection Loading...
        </span>
      </div>
    );
  }

  const isOnline = detectionStatus?.status === 'healthy';
  const method = detectionStatus?.method || 'Unknown';
  
  return (
    <div className="text-sm text-gray-500">
      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
        isOnline 
          ? 'bg-green-100 text-green-800' 
          : 'bg-red-100 text-red-800'
      }`}>
        {isOnline ? (
          <Eye className="inline w-2 h-2 mr-1" />
        ) : (
          <EyeOff className="inline w-2 h-2 mr-1" />
        )}
        {isOnline ? `Detection: ${method}` : 'Detection Offline'}
      </span>
    </div>
  );
}