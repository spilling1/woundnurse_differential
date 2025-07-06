import type { Express } from "express";
import { createServer, type Server } from "http";
import path from "path";
import { setupAuth } from "../replitAuth";
import { registerAuthRoutes } from "./auth-routes";
import { registerAssessmentRoutes } from "./assessment-routes";
import { registerFollowUpRoutes } from "./follow-up-routes";
import { registerAdminRoutes } from "./admin-routes";
import { spawn, ChildProcess } from "child_process";

// Global YOLO service management
let yoloProcess: ChildProcess | null = null;
let yoloHealthy = false;

function startYoloService() {
  if (yoloProcess) {
    try {
      yoloProcess.kill();
    } catch (e) {
      // Ignore error
    }
    yoloProcess = null;
  }

  console.log('Starting YOLO service...');
  yoloProcess = spawn('python3', ['yolo_smart_service.py'], {
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: false,
    cwd: process.cwd()
  });

  yoloProcess.stdout?.on('data', (data) => {
    console.log(`YOLO stdout: ${data}`);
  });

  yoloProcess.stderr?.on('data', (data) => {
    console.error(`YOLO stderr: ${data}`);
  });

  yoloProcess.on('exit', (code) => {
    console.log(`YOLO process exited with code ${code}`);
    yoloHealthy = false;
    yoloProcess = null;
    // Restart after 5 seconds
    setTimeout(() => {
      console.log('Restarting YOLO service...');
      startYoloService();
    }, 5000);
  });

  // Check health after startup
  setTimeout(async () => {
    try {
      const response = await fetch('http://localhost:8081/health');
      yoloHealthy = response.ok;
      console.log(`YOLO health check: ${yoloHealthy ? 'healthy' : 'failed'}`);
    } catch (error) {
      console.log('YOLO health check failed:', error);
      yoloHealthy = false;
    }
  }, 3000);
}

export async function registerRoutes(app: Express): Promise<Server> {
  // Start YOLO service
  startYoloService();
  
  // Auth middleware
  await setupAuth(app);

  // YOLO service proxy routes (accessible through port 5000)
  app.get('/api/yolo', async (req, res) => {
    try {
      const response = await fetch('http://localhost:8081/');
      const data = await response.json();
      res.json(data);
    } catch (error) {
      res.status(503).json({ error: 'YOLO service not available' });
    }
  });

  app.get('/api/yolo/health', async (req, res) => {
    try {
      const response = await fetch('http://localhost:8081/health');
      const data = await response.json();
      res.json(data);
    } catch (error) {
      res.status(503).json({ error: 'YOLO service not available' });
    }
  });

  // Detection status endpoint for UI status monitor
  app.get('/api/detection-status', async (req, res) => {
    // Check internal YOLO health status first
    if (yoloHealthy && yoloProcess) {
      res.json({
        status: 'healthy',
        method: 'YOLO',
        service: 'yolo-wound-detection',
        version: '1.0'
      });
      return;
    }
    
    // Double-check by trying to reach the service
    try {
      const yoloResponse = await fetch('http://localhost:8081/health');
      if (yoloResponse.ok) {
        const yoloData = await yoloResponse.json();
        yoloHealthy = true; // Update internal state
        res.json({
          status: 'healthy',
          method: 'YOLO',
          service: yoloData.service,
          version: yoloData.version
        });
        return;
      }
    } catch (error) {
      // YOLO not available, check cloud APIs
      console.log('YOLO service not available, checking cloud APIs');
    }

    // Check if cloud APIs are configured
    const hasGoogleApi = !!process.env.GOOGLE_API_KEY;
    const hasAzureApi = !!process.env.AZURE_COMPUTER_VISION_KEY;
    
    if (hasGoogleApi || hasAzureApi) {
      res.json({
        status: 'healthy',
        method: hasGoogleApi ? 'Google Cloud Vision' : 'Azure Computer Vision',
        service: 'cloud-detection',
        version: '1.0'
      });
    } else {
      res.json({
        status: 'healthy',
        method: 'Enhanced Fallback',
        service: 'fallback-detection',
        version: '1.0'
      });
    }
  });

  // Serve the YOLO test page
  app.get('/yolo-test', (req, res) => {
    res.send(`
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO Wound Detection Service Test</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .status {
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            font-weight: bold;
        }
        .success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .info {
            background-color: #cce7ff;
            color: #004085;
            border: 1px solid #b8daff;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }
        button:hover {
            background-color: #0056b3;
        }
        pre {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border: 1px solid #e9ecef;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🩺 YOLO Wound Detection Service</h1>
        
        <div class="info">
            <strong>Service Status:</strong> <span id="status">Checking...</span>
        </div>
        
        <div style="text-align: center; margin: 20px 0;">
            <button onclick="checkHealth()">Check Health</button>
            <button onclick="getServiceInfo()">Get Service Info</button>
        </div>
        
        <div id="result"></div>
        
        <script>
            async function checkHealth() {
                document.getElementById('status').textContent = 'Checking...';
                document.getElementById('result').innerHTML = '';
                
                try {
                    const response = await fetch('/api/yolo/health');
                    const data = await response.json();
                    
                    if (data.status === 'healthy') {
                        document.getElementById('status').textContent = 'Healthy ✅';
                        document.getElementById('result').innerHTML = \`
                            <div class="success">
                                <strong>YOLO Service is Running!</strong>
                                <pre>\${JSON.stringify(data, null, 2)}</pre>
                            </div>
                        \`;
                    } else {
                        throw new Error('Service not healthy');
                    }
                } catch (error) {
                    document.getElementById('status').textContent = 'Error ❌';
                    document.getElementById('result').innerHTML = \`
                        <div class="error">
                            <strong>Service Error:</strong> \${error.message}
                        </div>
                    \`;
                }
            }
            
            async function getServiceInfo() {
                document.getElementById('result').innerHTML = '';
                
                try {
                    const response = await fetch('/api/yolo');
                    const data = await response.json();
                    
                    document.getElementById('result').innerHTML = \`
                        <div class="success">
                            <strong>Service Information:</strong>
                            <pre>\${JSON.stringify(data, null, 2)}</pre>
                        </div>
                    \`;
                } catch (error) {
                    document.getElementById('result').innerHTML = \`
                        <div class="error">
                            <strong>Service Error:</strong> \${error.message}
                        </div>
                    \`;
                }
            }
            
            // Check health on page load
            window.onload = checkHealth;
        </script>
    </div>
</body>
</html>
    `);
  });

  // Public AI models endpoint for user dropdowns
  app.get('/api/ai-analysis-models', async (req, res) => {
    try {
      const { storage } = await import('../storage');
      const models = await storage.getEnabledAiAnalysisModels();
      res.json(models);
    } catch (error: any) {
      console.error('Error fetching enabled AI analysis models:', error);
      res.status(500).json({
        code: "FETCH_AI_MODELS_ERROR",
        message: error.message || "Failed to fetch AI models"
      });
    }
  });

  // Register all route modules
  registerAuthRoutes(app);
  registerAssessmentRoutes(app);
  registerFollowUpRoutes(app);
  registerAdminRoutes(app);

  const httpServer = createServer(app);
  return httpServer;
} 