/**
 * Body Region Mapping for the new numbered body diagrams
 * Coordinates are based on the actual numbered regions visible in the images
 */

export interface BodyRegion {
  id: string;
  name: string;
  description: string;
  coordinates: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  commonWoundTypes: string[];
}

// Front body regions matching the numbered regions in the front view diagram
export const frontBodyRegions: BodyRegion[] = [
  {
    id: '5',
    name: 'Abdomen',
    description: 'Abdominal region',
    coordinates: { x: 190, y: 385, width: 120, height: 60 },
    commonWoundTypes: ['surgical_wound', 'traumatic_wound']
  },
  {
    id: '6L',
    name: 'Left Upper Leg',
    description: 'Left upper leg/thigh',
    coordinates: { x: 140, y: 465, width: 80, height: 125 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '6R',
    name: 'Right Upper Leg',
    description: 'Right upper leg/thigh',
    coordinates: { x: 280, y: 465, width: 80, height: 125 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '7L',
    name: 'Left Lower Leg',
    description: 'Left lower leg/shin',
    coordinates: { x: 140, y: 590, width: 80, height: 125 },
    commonWoundTypes: ['venous_ulcer', 'traumatic_wound']
  },
  {
    id: '7R',
    name: 'Right Lower Leg',
    description: 'Right lower leg/shin',
    coordinates: { x: 280, y: 590, width: 80, height: 125 },
    commonWoundTypes: ['venous_ulcer', 'traumatic_wound']
  },
  {
    id: '8L',
    name: 'Left Foot',
    description: 'Left foot',
    coordinates: { x: 140, y: 715, width: 80, height: 65 },
    commonWoundTypes: ['diabetic_ulcer', 'pressure_injury']
  },
  {
    id: '8R',
    name: 'Right Foot',
    description: 'Right foot',
    coordinates: { x: 280, y: 715, width: 80, height: 65 },
    commonWoundTypes: ['diabetic_ulcer', 'pressure_injury']
  }
];

// Back body regions matching the numbered regions in the back view diagram
export const backBodyRegions: BodyRegion[] = [
  {
    id: '1',
    name: 'Lower Back',
    description: 'Lower back/sacral region',
    coordinates: { x: 190, y: 385, width: 120, height: 85 },
    commonWoundTypes: ['pressure_injury', 'surgical_wound']
  },
  {
    id: '2L',
    name: 'Left Posterior Thigh',
    description: 'Left posterior thigh',
    coordinates: { x: 140, y: 470, width: 80, height: 125 },
    commonWoundTypes: ['pressure_injury', 'traumatic_wound']
  },
  {
    id: '2R',
    name: 'Right Posterior Thigh',
    description: 'Right posterior thigh',
    coordinates: { x: 280, y: 470, width: 80, height: 125 },
    commonWoundTypes: ['pressure_injury', 'traumatic_wound']
  },
  {
    id: '3L',
    name: 'Left Posterior Calf',
    description: 'Left posterior calf',
    coordinates: { x: 140, y: 595, width: 80, height: 125 },
    commonWoundTypes: ['venous_ulcer', 'traumatic_wound']
  },
  {
    id: '3R',
    name: 'Right Posterior Calf',
    description: 'Right posterior calf',
    coordinates: { x: 280, y: 595, width: 80, height: 125 },
    commonWoundTypes: ['venous_ulcer', 'traumatic_wound']
  },
  {
    id: '4L',
    name: 'Left Heel',
    description: 'Left heel',
    coordinates: { x: 140, y: 720, width: 80, height: 65 },
    commonWoundTypes: ['pressure_injury', 'diabetic_ulcer']
  },
  {
    id: '4R',
    name: 'Right Heel',
    description: 'Right heel',
    coordinates: { x: 280, y: 720, width: 80, height: 65 },
    commonWoundTypes: ['pressure_injury', 'diabetic_ulcer']
  }
];

// Helper functions to get regions by view
export function getBodyRegionsByImage(view: 'front' | 'back'): BodyRegion[] {
  return view === 'front' ? frontBodyRegions : backBodyRegions;
}

export function getBodyRegionByCoordinates(x: number, y: number, view: 'front' | 'back'): BodyRegion | null {
  const regions = getBodyRegionsByImage(view);
  return regions.find(region => {
    const { coordinates } = region;
    return (
      x >= coordinates.x &&
      x <= coordinates.x + coordinates.width &&
      y >= coordinates.y &&
      y <= coordinates.y + coordinates.height
    );
  }) || null;
}

export function getBodyRegionById(id: string, view: 'front' | 'back'): BodyRegion | null {
  const regions = getBodyRegionsByImage(view);
  return regions.find(region => region.id === id) || null;
}