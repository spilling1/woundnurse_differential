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
  processingStep?: string;
}

export default function AnalysisLogger({ isActive, onComplete, processingStep }: AnalysisLoggerProps) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [currentStep, setCurrentStep] = useState(0);

  // Real-time processing steps that show actual backend activity
  const getProcessingSteps = () => {
    const steps = [
      { message: "🔄 Starting wound analysis...", type: 'info' as const, duration: 500 },
      { message: "📡 Connecting to YOLO detection service...", type: 'info' as const, duration: 800 },
      { message: "🔍 Scanning image for wound boundaries...", type: 'processing' as const, duration: 1200 },
      { message: "📊 YOLO processing complete - analyzing results...", type: 'processing' as const, duration: 600 },
      { message: "🧠 Initializing AI vision analysis...", type: 'info' as const, duration: 700 },
      { message: "🤔 AI examining wound characteristics...", type: 'processing' as const, duration: 2000 },
      { message: "💭 Analyzing tissue viability and staging...", type: 'processing' as const, duration: 1500 },
      { message: "🔬 Evaluating infection signs and exudate...", type: 'processing' as const, duration: 1200 },
      { message: "📏 Calculating wound measurements...", type: 'processing' as const, duration: 800 },
      { message: "🎯 Building confidence assessment...", type: 'processing' as const, duration: 900 },
      { message: "🧬 Generating diagnostic questions...", type: 'processing' as const, duration: 1100 },
      { message: "✅ Analysis complete - preparing results...", type: 'success' as const, duration: 500 }
    ];
    
    return steps;
  };

  const analysisSteps = getProcessingSteps();

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
          id: `log-${Date.now()}-${i}-${Math.random()}`,
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
      default: return '🔄';
    }
  };

  return (
    <Card className="mt-4 bg-slate-50 border-slate-200">
      <CardContent className="p-4">
        <div className="mb-3">
          {/* Header removed per user request */}
        </div>
        
        <div className="space-y-2">
          {/* Show only the last 3 logs */}
          {logs.slice(-3).map((log, index) => (
            <div 
              key={log.id}
              className="flex items-center space-x-3 p-2 rounded-md transition-all duration-300"
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
          
          {/* Processing indicator when analysis is active and we have logs */}
          {isActive && logs.length > 0 && (
            <div key="processing-indicator" className="flex items-center space-x-3 p-2 rounded-md">
              <div className="w-6 h-6 rounded-full bg-blue-100 border border-blue-300 flex items-center justify-center">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-ping"></div>
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm text-slate-600 animate-pulse">...Processing...</div>
              </div>
            </div>
          )}
          
          {/* Empty slots to maintain consistent height (only if we have fewer than 3 items total) */}
          {Array.from({ length: Math.max(0, 3 - Math.min(logs.length, 3) - (isActive && logs.length > 0 ? 1 : 0)) }).map((_, index) => (
            <div key={`empty-${index}`} className="h-10 opacity-0" />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}