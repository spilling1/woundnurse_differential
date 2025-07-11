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

// Front body regions matching the layout shown in the user's image
export const frontBodyRegions: BodyRegion[] = [
  // Numbered regions from the diagram
  {
    id: '5',
    name: 'Stomach',
    description: 'Abdominal/stomach region',
    coordinates: { x: 185, y: 365, width: 130, height: 75 },
    commonWoundTypes: ['surgical_wound', 'traumatic_wound']
  },
  {
    id: '6L',
    name: 'Left Upper Leg (Front)',
    description: 'Left upper leg/thigh front',
    coordinates: { x: 140, y: 440, width: 80, height: 140 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '6R',
    name: 'Right Upper Leg (Front)',
    description: 'Right upper leg/thigh front',
    coordinates: { x: 280, y: 440, width: 80, height: 140 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '7L',
    name: 'Left Lower Leg (Front)',
    description: 'Left lower leg/shin front',
    coordinates: { x: 140, y: 580, width: 80, height: 140 },
    commonWoundTypes: ['venous_ulcer', 'traumatic_wound']
  },
  {
    id: '7R',
    name: 'Right Lower Leg (Front)',
    description: 'Right lower leg/shin front',
    coordinates: { x: 280, y: 580, width: 80, height: 140 },
    commonWoundTypes: ['venous_ulcer', 'traumatic_wound']
  },
  {
    id: '8L',
    name: 'Left Top/Side of Foot',
    description: 'Left top/side of foot',
    coordinates: { x: 140, y: 720, width: 80, height: 80 },
    commonWoundTypes: ['diabetic_ulcer', 'pressure_injury']
  },
  {
    id: '8R',
    name: 'Right Top/Side of Foot',
    description: 'Right top/side of foot',
    coordinates: { x: 280, y: 720, width: 80, height: 80 },
    commonWoundTypes: ['diabetic_ulcer', 'pressure_injury']
  },
  
  // Additional unlabeled regions
  {
    id: 'head_neck_front',
    name: 'Head/Neck (Front)',
    description: 'Head and neck area - front view',
    coordinates: { x: 190, y: 10, width: 120, height: 100 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: 'chest',
    name: 'Chest',
    description: 'Chest area',
    coordinates: { x: 185, y: 110, width: 130, height: 120 },
    commonWoundTypes: ['surgical_wound', 'traumatic_wound']
  },
  {
    id: 'arms_front_left',
    name: 'Left Arm (Front)',
    description: 'Left arm - front view',
    coordinates: { x: 35, y: 110, width: 140, height: 290 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: 'arms_front_right',
    name: 'Right Arm (Front)',
    description: 'Right arm - front view',
    coordinates: { x: 325, y: 110, width: 140, height: 290 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  }
];

// Back body regions matching the layout shown in the user's image
export const backBodyRegions: BodyRegion[] = [
  // Numbered regions from the diagram
  {
    id: '1',
    name: 'Buttox',
    description: 'Buttox/gluteal region',
    coordinates: { x: 185, y: 365, width: 130, height: 100 },
    commonWoundTypes: ['pressure_injury', 'surgical_wound']
  },
  {
    id: '2L',
    name: 'Left Upper Leg (Back)',
    description: 'Left upper leg/thigh back',
    coordinates: { x: 140, y: 465, width: 80, height: 140 },
    commonWoundTypes: ['pressure_injury', 'traumatic_wound']
  },
  {
    id: '2R',
    name: 'Right Upper Leg (Back)',
    description: 'Right upper leg/thigh back',
    coordinates: { x: 280, y: 465, width: 80, height: 140 },
    commonWoundTypes: ['pressure_injury', 'traumatic_wound']
  },
  {
    id: '3L',
    name: 'Left Lower Leg (Back)',
    description: 'Left lower leg/calf back',
    coordinates: { x: 140, y: 605, width: 80, height: 140 },
    commonWoundTypes: ['venous_ulcer', 'traumatic_wound']
  },
  {
    id: '3R',
    name: 'Right Lower Leg (Back)',
    description: 'Right lower leg/calf back',
    coordinates: { x: 280, y: 605, width: 80, height: 140 },
    commonWoundTypes: ['venous_ulcer', 'traumatic_wound']
  },
  {
    id: '4L',
    name: 'Left Heel/Sole of Foot',
    description: 'Left heel/sole of foot',
    coordinates: { x: 140, y: 745, width: 80, height: 65 },
    commonWoundTypes: ['pressure_injury', 'diabetic_ulcer']
  },
  {
    id: '4R',
    name: 'Right Heel/Sole of Foot',
    description: 'Right heel/sole of foot',
    coordinates: { x: 280, y: 745, width: 80, height: 65 },
    commonWoundTypes: ['pressure_injury', 'diabetic_ulcer']
  },
  
  // Additional unlabeled regions
  {
    id: 'head_neck_back',
    name: 'Head/Neck (Back)',
    description: 'Head and neck area - back view',
    coordinates: { x: 190, y: 10, width: 120, height: 100 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: 'back',
    name: 'Back',
    description: 'Upper back area',
    coordinates: { x: 185, y: 110, width: 130, height: 130 },
    commonWoundTypes: ['pressure_injury', 'surgical_wound']
  },
  {
    id: 'arms_back_left',
    name: 'Left Arm (Back)',
    description: 'Left arm - back view',
    coordinates: { x: 35, y: 110, width: 140, height: 290 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: 'arms_back_right',
    name: 'Right Arm (Back)',
    description: 'Right arm - back view',
    coordinates: { x: 325, y: 110, width: 140, height: 290 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
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