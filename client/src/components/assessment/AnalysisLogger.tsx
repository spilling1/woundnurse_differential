import { useState, useEffect } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface LogEntry {
  id: string;
  timestamp: string;
  message: string;
  type: 'info' | 'success' | 'processing' | 'warning';
  duration?: number;
}

interface AnalysisLoggerProps {
  isActive: boolean;
  onComplete?: () => void;
}

export default function AnalysisLogger({ isActive, onComplete }: AnalysisLoggerProps) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [currentStep, setCurrentStep] = useState(0);

  // Predefined analysis steps that simulate real processing
  const analysisSteps = [
    { message: "Initializing YOLO detection service...", type: 'info' as const, duration: 800 },
    { message: "Loading wound detection model (YOLOv8)...", type: 'info' as const, duration: 1200 },
    { message: "Preprocessing image for detection...", type: 'processing' as const, duration: 600 },
    { message: "Running YOLO wound boundary detection...", type: 'processing' as const, duration: 1400 },
    { message: "Analyzing wound characteristics and measurements...", type: 'processing' as const, duration: 900 },
    { message: "YOLO detection complete - processing results...", type: 'success' as const, duration: 500 },
    { message: "Initializing AI vision analysis...", type: 'info' as const, duration: 700 },
    { message: "Sending image to Gemini 2.5 Pro model...", type: 'info' as const, duration: 1000 },
    { message: "AI analyzing wound type and characteristics...", type: 'processing' as const, duration: 2500 },
    { message: "Evaluating tissue viability and stage...", type: 'processing' as const, duration: 1800 },
    { message: "Assessing infection signs and exudate...", type: 'processing' as const, duration: 1400 },
    { message: "Calculating wound size and measurements...", type: 'processing' as const, duration: 1100 },
    { message: "Generating confidence assessment...", type: 'processing' as const, duration: 800 },
    { message: "Fusing YOLO and AI analysis results...", type: 'processing' as const, duration: 600 },
    { message: "Finalizing wound classification...", type: 'success' as const, duration: 500 },
    { message: "Analysis complete - preparing assessment...", type: 'success' as const, duration: 400 }
  ];

  useEffect(() => {
    if (!isActive) {
      setLogs([]);
      setCurrentStep(0);
      return;
    }

    const runAnalysis = async () => {
      for (let i = 0; i < analysisSteps.length; i++) {
        const step = analysisSteps[i];
        const logEntry: LogEntry = {
          id: `log-${Date.now()}-${i}`,
          timestamp: new Date().toLocaleTimeString(),
          message: step.message,
          type: step.type
        };

        setLogs(prev => {
          const newLogs = [...prev, logEntry];
          // Keep only the last 3 entries
          return newLogs.slice(-3);
        });

        setCurrentStep(i + 1);

        // Wait for step duration
        await new Promise(resolve => setTimeout(resolve, step.duration));
      }

      // Complete the analysis
      setTimeout(() => {
        onComplete?.();
      }, 500);
    };

    runAnalysis();
  }, [isActive, onComplete]);

  if (!isActive) return null;

  const getTypeColor = (type: LogEntry['type']) => {
    switch (type) {
      case 'success': return 'bg-green-100 text-green-800 border-green-200';
      case 'processing': return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'warning': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getTypeIcon = (type: LogEntry['type']) => {
    switch (type) {
      case 'success': return '✓';
      case 'processing': return '⚡';
      case 'warning': return '⚠';
      default: return '→';
    }
  };

  return (
    <Card className="mt-4 bg-slate-50 border-slate-200">
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-slate-700">Analysis Progress</h3>
          <Badge variant="outline" className="text-xs">
            {currentStep}/{analysisSteps.length}
          </Badge>
        </div>
        
        <div className="space-y-2">
          {logs.map((log, index) => (
            <div 
              key={log.id}
              className={`flex items-center space-x-3 p-2 rounded-md transition-all duration-300 ${
                index === logs.length - 1 ? 'animate-pulse' : ''
              }`}
            >
              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium ${getTypeColor(log.type)}`}>
                {getTypeIcon(log.type)}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm text-slate-700 truncate">{log.message}</div>
              </div>
              <div className="text-xs text-slate-500 font-mono">
                {log.timestamp}
              </div>
            </div>
          ))}
          
          {/* Empty slots to maintain consistent height */}
          {Array.from({ length: 3 - logs.length }).map((_, index) => (
            <div key={`empty-${index}`} className="h-10 opacity-0" />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}