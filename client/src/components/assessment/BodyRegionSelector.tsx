import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  getBodyRegionsByImage, 
  getBodyRegionByCoordinates,
  BodyRegion 
} from './BodyRegionMapping';

// Import the body diagram images
// Use the new body diagram images
const FrontBodyPath = '/FrontBody_1752200712809.png';
const BackBodyPath = '/BackBody_1752200448378.png';

interface BodyRegionSelectorProps {
  selectedRegion: { id: string; name: string } | null;
  onRegionSelect: (region: { id: string; name: string } | null) => void;
}

export default function BodyRegionSelector({ selectedRegion, onRegionSelect }: BodyRegionSelectorProps) {
  const [currentView, setCurrentView] = useState<'front' | 'back'>('front');
  const [hoveredRegion, setHoveredRegion] = useState<BodyRegion | null>(null);
  const [showRegionOutlines, setShowRegionOutlines] = useState(true);

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
        onRegionSelect({ id: region.id, name: region.name });
      }
    }
  };

  const handleImageHover = (event: React.MouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    // Convert to absolute coordinates based on the image dimensions
    const x = (event.clientX - rect.left) / rect.width * 500; // Image width
    const y = (event.clientY - rect.top) / rect.height * 820; // Image height
    
    const region = getBodyRegionByCoordinates(x, y, currentView);
    setHoveredRegion(region || null);
  };

  const handleImageLeave = () => {
    setHoveredRegion(null);
  };

  const renderRegionOverlays = () => {
    if (!showRegionOutlines) return null;
    
    const regions = getBodyRegionsByImage(currentView);
    
    return (
      <div className="absolute inset-0 w-full h-full pointer-events-none">
        {regions.map(region => {
          const isSelected = selectedRegion?.id === region.id;
          const isHovered = hoveredRegion?.id === region.id;
          
          return (
            <div
              key={region.id}
              className={`absolute border-2 transition-all duration-200 ${
                isSelected 
                  ? 'bg-blue-500 bg-opacity-40 border-blue-600' 
                  : isHovered 
                    ? 'bg-green-500 bg-opacity-30 border-green-600' 
                    : 'bg-gray-500 bg-opacity-10 border-gray-400'
              }`}
              style={{
                left: `${(region.coordinates.x / 500) * 100}%`,
                top: `${(region.coordinates.y / 820) * 100}%`,
                width: `${(region.coordinates.width / 500) * 100}%`,
                height: `${(region.coordinates.height / 820) * 100}%`,
              }}
            >
              <div className={`w-full h-full flex items-center justify-center text-xs font-bold ${
                isSelected 
                  ? 'text-blue-900' 
                  : isHovered 
                    ? 'text-green-900' 
                    : 'text-gray-600'
              }`}>
                {region.id}
              </div>
            </div>
          );
        })}
      </div>
    );
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
            variant={showRegionOutlines ? 'default' : 'outline'}
            onClick={() => setShowRegionOutlines(!showRegionOutlines)}
            size="sm"
          >
            Show Outlines
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
            {showRegionOutlines && renderRegionOverlays()}
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
              <span className="font-medium text-green-900">Hover: {hoveredRegion.name}</span>
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
            <li>• Click "Show Outlines" to display numbered clickable regions</li>
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