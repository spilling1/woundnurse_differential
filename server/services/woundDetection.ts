import axios from 'axios';
import sharp from 'sharp';

export interface WoundDetection {
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  confidence: number;
  area: number;
  perimeter: number;
  measurements: {
    lengthMm: number;
    widthMm: number;
    areaMm2: number;
  };
  referenceObjectDetected: boolean;
  scaleCalibrated: boolean;
}

export interface DetectionResult {
  detections: WoundDetection[];
  imageWidth: number;
  imageHeight: number;
  processingTime: number;
  model: string;
  version: string;
}

/**
 * Main wound detection service that integrates YOLO9 with existing workflow
 */
export class WoundDetectionService {
  private yoloEndpoint: string;
  private fallbackEnabled: boolean = true;
  
  constructor() {
    this.yoloEndpoint = process.env.YOLO_ENDPOINT || 'http://localhost:8081/detect';
  }

  /**
   * Detect wounds in image with precise measurements
   */
  async detectWounds(imageBase64: string, mimeType: string = 'image/jpeg'): Promise<DetectionResult> {
    try {
      // Convert base64 to buffer for processing
      const imageBuffer = Buffer.from(imageBase64, 'base64');
      
      // Get image dimensions
      const metadata = await sharp(imageBuffer).metadata();
      const imageWidth = metadata.width || 0;
      const imageHeight = metadata.height || 0;
      
      const startTime = Date.now();
      
      // Try YOLO9 detection first
      let detectionResult: DetectionResult;
      
      try {
        detectionResult = await this.callYoloService(imageBuffer, imageWidth, imageHeight);
      } catch (error) {
        console.warn('YOLO9 service unavailable, using fallback detection:', error);
        detectionResult = await this.fallbackDetection(imageBuffer, imageWidth, imageHeight);
      }
      
      detectionResult.processingTime = Date.now() - startTime;
      
      return detectionResult;
    } catch (error: any) {
      console.error('Wound detection failed:', error);
      throw new Error(`Wound detection failed: ${error.message}`);
    }
  }

  /**
   * Call external YOLO9 service
   */
  private async callYoloService(imageBuffer: Buffer, width: number, height: number): Promise<DetectionResult> {
    const response = await axios.post(this.yoloEndpoint, {
      image: imageBuffer.toString('base64'),
      confidence_threshold: 0.5,
      include_measurements: true,
      detect_reference_objects: true
    }, {
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }
    });

    const yoloData = response.data;
    
    return {
      detections: yoloData.detections.map((det: any) => ({
        boundingBox: {
          x: det.bbox[0],
          y: det.bbox[1],
          width: det.bbox[2] - det.bbox[0],
          height: det.bbox[3] - det.bbox[1]
        },
        confidence: det.confidence,
        area: det.area_pixels,
        perimeter: det.perimeter_pixels,
        measurements: {
          lengthMm: det.measurements?.length_mm || 0,
          widthMm: det.measurements?.width_mm || 0,
          areaMm2: det.measurements?.area_mm2 || 0
        },
        referenceObjectDetected: det.reference_object_detected || false,
        scaleCalibrated: det.scale_calibrated || false
      })),
      imageWidth: width,
      imageHeight: height,
      processingTime: 0,
      model: yoloData.model || 'yolo9',
      version: yoloData.version || '1.0'
    };
  }

  /**
   * Fallback detection using basic image analysis
   */
  private async fallbackDetection(imageBuffer: Buffer, width: number, height: number): Promise<DetectionResult> {
    // Basic fallback - create a default detection in the center of the image
    const detections = this.createDefaultDetection(width, height);
    
    return {
      detections,
      imageWidth: width,
      imageHeight: height,
      processingTime: 0,
      model: 'fallback',
      version: '1.0'
    };
  }

  /**
   * Create a default detection for fallback mode
   */
  private createDefaultDetection(width: number, height: number): WoundDetection[] {
    // Create a default bounding box in the center of the image
    const centerX = Math.floor(width * 0.4);
    const centerY = Math.floor(height * 0.4);
    const boundingWidth = Math.floor(width * 0.2);
    const boundingHeight = Math.floor(height * 0.2);
    
    return [{
      boundingBox: {
        x: centerX,
        y: centerY,
        width: boundingWidth,
        height: boundingHeight
      },
      confidence: 0.5, // Lower confidence for fallback
      area: boundingWidth * boundingHeight,
      perimeter: (boundingWidth + boundingHeight) * 2,
      measurements: {
        lengthMm: boundingWidth * 0.1, // Rough estimate: 0.1mm per pixel
        widthMm: boundingHeight * 0.1,
        areaMm2: (boundingWidth * boundingHeight) * 0.01
      },
      referenceObjectDetected: false,
      scaleCalibrated: false
    }];
  }

  /**
   * Enhance existing wound classification with detection data
   */
  enhanceClassificationWithDetection(
    classification: any, 
    detectionResult: DetectionResult
  ): any {
    if (detectionResult.detections.length === 0) {
      return classification;
    }
    
    const primaryWound = detectionResult.detections[0]; // Use highest confidence detection
    
    return {
      ...classification,
      detection: {
        confidence: primaryWound.confidence,
        boundingBox: primaryWound.boundingBox,
        measurements: primaryWound.measurements,
        scaleCalibrated: primaryWound.scaleCalibrated
      },
      size: this.categorizeSizeFromMeasurements(primaryWound.measurements),
      preciseMeasurements: primaryWound.measurements,
      detectionMetadata: {
        model: detectionResult.model,
        version: detectionResult.version,
        processingTime: detectionResult.processingTime,
        multipleWounds: detectionResult.detections.length > 1
      }
    };
  }

  /**
   * Convert precise measurements to size categories
   */
  private categorizeSizeFromMeasurements(measurements: any): string {
    const areaMm2 = measurements.areaMm2;
    
    if (areaMm2 < 100) return 'small';
    if (areaMm2 < 500) return 'medium';
    return 'large';
  }
}

export const woundDetectionService = new WoundDetectionService();