import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import FrontBodyPath from '@assets/FrontBody_1752196192722.png';
import BackBodyPath from '@assets/BackBody_1752196189291.png';
import { 
  getBodyRegionsSVGByImage, 
  getBodyRegionSVGByCoordinates,
  BodyRegionSVG 
} from './BodyRegionSVG';

interface BodyRegionSelectorProps {
  selectedRegion: { id: string; name: string } | null;
  onRegionSelect: (region: { id: string; name: string } | null) => void;
}

export default function BodyRegionSelector({ selectedRegion, onRegionSelect }: BodyRegionSelectorProps) {
  const [currentView, setCurrentView] = useState<'front' | 'back'>('front');
  const [hoveredRegion, setHoveredRegion] = useState<BodyRegionSVG | null>(null);
  const [showRegionOutlines, setShowRegionOutlines] = useState(true);

  const handleImageClick = (event: React.MouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    // Convert to absolute coordinates based on the image dimensions
    const x = (event.clientX - rect.left) / rect.width * 645; // Image width
    const y = (event.clientY - rect.top) / rect.height * 400; // Image height
    
    const region = getBodyRegionSVGByCoordinates(x, y, currentView);
    
    if (region) {
      if (selectedRegion?.id === region.id) {
        // Deselect if clicking the same region
        onRegionSelect(null);
      } else {
        // Select the new region
        onRegionSelect({ id: region.id, name: region.name });
      }
    }
  };

  const handleImageHover = (event: React.MouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    // Convert to absolute coordinates based on the image dimensions
    const x = (event.clientX - rect.left) / rect.width * 645; // Image width
    const y = (event.clientY - rect.top) / rect.height * 400; // Image height
    
    const region = getBodyRegionSVGByCoordinates(x, y, currentView);
    setHoveredRegion(region || null);
  };

  const handleImageLeave = () => {
    setHoveredRegion(null);
  };

  const renderSVGOverlays = () => {
    if (!showRegionOutlines) return null;
    
    const regions = getBodyRegionsSVGByImage(currentView);
    
    return (
      <svg
        className="absolute inset-0 w-full h-full pointer-events-none"
        viewBox="0 0 645 400"
        preserveAspectRatio="xMidYMid meet"
      >
        {regions.map(region => {
          const isSelected = selectedRegion?.id === region.id;
          const isHovered = hoveredRegion?.id === region.id;
          
          return (
            <g key={region.id}>
              <path
                d={region.svgPath}
                fill={
                  isSelected 
                    ? 'rgba(59, 130, 246, 0.4)' 
                    : isHovered 
                      ? 'rgba(34, 197, 94, 0.3)' 
                      : 'rgba(156, 163, 175, 0.1)'
                }
                stroke={
                  isSelected 
                    ? '#3b82f6' 
                    : isHovered 
                      ? '#22c55e' 
                      : '#9ca3af'
                }
                strokeWidth="2"
                className="transition-all duration-200"
              />
              {/* Add region number label */}
              <text
                x={getPathCenterX(region.svgPath)}
                y={getPathCenterY(region.svgPath)}
                textAnchor="middle"
                dominantBaseline="middle"
                className={`text-xs font-bold select-none ${
                  isSelected 
                    ? 'fill-blue-800' 
                    : isHovered 
                      ? 'fill-green-800' 
                      : 'fill-gray-600'
                }`}
              >
                {region.id}
              </text>
            </g>
          );
        })}
      </svg>
    );
  };

  // Helper functions to get path center (simplified)
  const getPathCenterX = (path: string): number => {
    const matches = path.match(/M\s*(\d+)\s*(\d+).*?Q\s*(\d+)\s*(\d+)/);
    if (matches) {
      const x1 = parseInt(matches[1]);
      const x2 = parseInt(matches[3]);
      return (x1 + x2) / 2;
    }
    return 0;
  };

  const getPathCenterY = (path: string): number => {
    const matches = path.match(/M\s*(\d+)\s*(\d+).*?Q\s*(\d+)\s*(\d+)/);
    if (matches) {
      const y1 = parseInt(matches[2]);
      const y2 = parseInt(matches[4]);
      return (y1 + y2) / 2;
    }
    return 0;
  };

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle className="text-xl font-semibold text-center">
          Select Body Region
        </CardTitle>
        <p className="text-sm text-gray-600 text-center">
          Click on the body diagram to identify where the wound is located
        </p>
      </CardHeader>
      <CardContent>
        {/* View toggle buttons */}
        <div className="flex justify-center gap-2 mb-4">
          <Button
            variant={currentView === 'front' ? 'default' : 'outline'}
            onClick={() => setCurrentView('front')}
            size="sm"
          >
            Front View
          </Button>
          <Button
            variant={currentView === 'back' ? 'default' : 'outline'}
            onClick={() => setCurrentView('back')}
            size="sm"
          >
            Back View
          </Button>
          <Button
            variant={showRegionOutlines ? 'default' : 'outline'}
            onClick={() => setShowRegionOutlines(!showRegionOutlines)}
            size="sm"
          >
            Show Outlines
          </Button>
        </div>

        {/* Body diagram with SVG overlays */}
        <div className="relative mx-auto max-w-lg">
          <div
            className="relative cursor-pointer"
            onClick={handleImageClick}
            onMouseMove={handleImageHover}
            onMouseLeave={handleImageLeave}
          >
            <img
              src={currentView === 'front' ? FrontBodyPath : BackBodyPath}
              alt={`${currentView} body diagram`}
              className="w-full h-auto"
            />
            {renderSVGOverlays()}
          </div>
        </div>

        {/* Region information display */}
        <div className="mt-4 space-y-2">
          {selectedRegion && (
            <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-medium text-blue-900">Selected Region:</h4>
                  <p className="text-sm text-blue-700">{selectedRegion.name}</p>
                </div>
                <Badge variant="secondary">#{selectedRegion.id}</Badge>
              </div>
            </div>
          )}
          
          {hoveredRegion && hoveredRegion.id !== selectedRegion?.id && (
            <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-medium text-green-900">Hover:</h4>
                  <p className="text-sm text-green-700">{hoveredRegion.name}</p>
                </div>
                <Badge variant="outline">#{hoveredRegion.id}</Badge>
              </div>
            </div>
          )}
        </div>

        {/* Instructions */}
        <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
          <h4 className="font-medium text-amber-900 mb-2">Instructions:</h4>
          <ul className="text-sm text-amber-800 space-y-1">
            <li>• Click "Show Outlines" to display body region outlines</li>
            <li>• Click on any outlined region to select the wound location</li>
            <li>• Switch between front and back views using the buttons above</li>
            <li>• The selected region will help our AI provide more accurate assessment</li>
            <li>• You can change your selection at any time</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}