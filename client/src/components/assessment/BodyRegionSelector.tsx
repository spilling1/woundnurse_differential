import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { bodyRegions, getBodyRegionById, getBodyRegionsByImage, type BodyRegion } from './shared/BodyRegionMapping';

// Import the body diagram images
import FrontBodyPath from '@assets/FrontBody_1752196192722.png';
import BackBodyPath from '@assets/BackBody_1752196189291.png';

interface BodyRegionSelectorProps {
  selectedRegion: { id: string; name: string } | null;
  onRegionSelect: (region: { id: string; name: string } | null) => void;
}

export default function BodyRegionSelector({ selectedRegion, onRegionSelect }: BodyRegionSelectorProps) {
  const [currentView, setCurrentView] = useState<'front' | 'back'>('front');
  const [hoveredRegion, setHoveredRegion] = useState<BodyRegion | null>(null);
  const [showRegionNumbers, setShowRegionNumbers] = useState(true);

  const handleImageClick = (event: React.MouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    // Convert to absolute coordinates based on the image dimensions
    const x = (event.clientX - rect.left) / rect.width * 645; // Image width
    const y = (event.clientY - rect.top) / rect.height * 400; // Image height
    
    const region = getBodyRegionByCoordinates(x, y, currentView);
    
    if (region) {
      if (selectedRegion?.id === region.id) {
        // Deselect if clicking the same region
        onRegionSelect(null);
      } else {
        // Select the new region
        onRegionSelect({ id: region.id, name: region.displayName });
      }
    }
  };

  const getBodyRegionByCoordinates = (x: number, y: number, view: 'front' | 'back'): BodyRegion | undefined => {
    const regions = getBodyRegionsByImage(view);
    
    return regions.find(region => {
      const { coordinates } = region;
      return (
        x >= coordinates.x &&
        x <= coordinates.x + coordinates.width &&
        y >= coordinates.y &&
        y <= coordinates.y + coordinates.height
      );
    });
  };

  const handleImageHover = (event: React.MouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    // Convert to absolute coordinates based on the image dimensions
    const x = (event.clientX - rect.left) / rect.width * 645; // Image width
    const y = (event.clientY - rect.top) / rect.height * 400; // Image height
    
    const region = getBodyRegionByCoordinates(x, y, currentView);
    setHoveredRegion(region || null);
  };

  const handleImageLeave = () => {
    setHoveredRegion(null);
  };

  const renderRegionOverlays = () => {
    const regions = getBodyRegionsByImage(currentView);
    
    return regions.map(region => {
      const isSelected = selectedRegion?.id === region.id;
      const isHovered = hoveredRegion?.id === region.id;
      
      // Convert absolute coordinates to percentages for CSS positioning
      const leftPercent = (region.coordinates.x / 645) * 100; // 645 is image width
      const topPercent = (region.coordinates.y / 400) * 100; // 400 is image height
      const widthPercent = (region.coordinates.width / 645) * 100;
      const heightPercent = (region.coordinates.height / 400) * 100;
      
      // Extract the number from the region ID (e.g., "272" from "272")
      const regionNumber = region.id.replace(/\D/g, '');
      
      return (
        <div
          key={region.id}
          className={`absolute border-2 rounded-lg transition-all duration-200 pointer-events-none flex items-center justify-center ${
            isSelected 
              ? 'border-blue-500 bg-blue-200/40' 
              : isHovered 
                ? 'border-green-400 bg-green-200/30' 
                : 'border-gray-400/20 bg-gray-100/10'
          }`}
          style={{
            left: `${leftPercent}%`,
            top: `${topPercent}%`,
            width: `${widthPercent}%`,
            height: `${heightPercent}%`,
            minWidth: '20px',
            minHeight: '20px',
          }}
        >
          <span className={`text-xs font-bold select-none ${
            isSelected 
              ? 'text-blue-800' 
              : isHovered 
                ? 'text-green-800' 
                : 'text-gray-600'
          }`}>
            {regionNumber}
          </span>
        </div>
      );
    });
  };

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle className="text-center">Select Wound Location</CardTitle>
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
            variant={showRegionNumbers ? 'default' : 'outline'}
            onClick={() => setShowRegionNumbers(!showRegionNumbers)}
            size="sm"
          >
            Show Numbers
          </Button>
        </div>

        {/* Body diagram container */}
        <div className="relative mx-auto max-w-md">
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
            {showRegionNumbers && renderRegionOverlays()}
          </div>
        </div>

        {/* Region information display */}
        <div className="mt-4 space-y-2">
          {selectedRegion && (
            <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <span className="font-medium text-blue-900">Selected: {selectedRegion.name}</span>
                  <Badge variant="secondary" className="ml-2">
                    {currentView} view
                  </Badge>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onRegionSelect(null)}
                  className="text-blue-600 hover:text-blue-800"
                >
                  Clear
                </Button>
              </div>
            </div>
          )}
          
          {hoveredRegion && !selectedRegion && (
            <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
              <span className="font-medium text-green-900">Hover: {hoveredRegion.displayName}</span>
              <Badge variant="outline" className="ml-2">
                Click to select
              </Badge>
            </div>
          )}
          
          {!selectedRegion && !hoveredRegion && (
            <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg text-center">
              <span className="text-gray-600">No region selected - click on the body diagram above</span>
            </div>
          )}
        </div>

        {/* Instructions */}
        <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
          <h4 className="font-medium text-amber-900 mb-2">Instructions:</h4>
          <ul className="text-sm text-amber-800 space-y-1">
            <li>• Click "Show Numbers" to display numbered clickable regions</li>
            <li>• Click on any numbered region to select the wound location</li>
            <li>• Switch between front and back views using the buttons above</li>
            <li>• The selected region will help our AI provide more accurate assessment</li>
            <li>• You can change your selection at any time</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}