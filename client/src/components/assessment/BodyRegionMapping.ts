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
    id: 'torso',
    name: 'Torso',
    description: 'Torso/mid-body region',
    coordinates: { x: 150, y: 210, width: 190, height: 90 },
    commonWoundTypes: ['surgical_wound', 'traumatic_wound']
  },
  {
    id: '5',
    name: 'Stomach',
    description: 'Abdominal/stomach region',
    coordinates: { x: 150, y: 300, width: 190, height: 115 },
    commonWoundTypes: ['surgical_wound', 'traumatic_wound']
  },
  {
    id: '6L',
    name: 'Left Upper Leg (Front)',
    description: 'Left upper leg/thigh front',
    coordinates: { x: 250, y: 415, width: 80, height: 160 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '6R',
    name: 'Right Upper Leg (Front)',
    description: 'Right upper leg/thigh front',
    coordinates: { x: 150, y: 415, width: 80, height: 160 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '7L',
    name: 'Left Lower Leg (Front)',
    description: 'Left lower leg/shin front',
    coordinates: { x: 250, y: 575, width: 80, height: 140 },
    commonWoundTypes: ['venous_ulcer', 'traumatic_wound']
  },
  {
    id: '7R',
    name: 'Right Lower Leg (Front)',
    description: 'Right lower leg/shin front',
    coordinates: { x: 170, y: 575, width: 80, height: 140 },
    commonWoundTypes: ['venous_ulcer', 'traumatic_wound']
  },
  {
    id: '8L',
    name: 'Left Top/Side of Foot',
    description: 'Left top/side of foot',
    coordinates: { x: 250, y: 715, width: 80, height: 80 },
    commonWoundTypes: ['diabetic_ulcer', 'pressure_injury']
  },
  {
    id: '8R',
    name: 'Right Top/Side of Foot',
    description: 'Right top/side of foot',
    coordinates: { x: 170, y: 715, width: 80, height: 80 },
    commonWoundTypes: ['diabetic_ulcer', 'pressure_injury']
  },
  
  // Additional unlabeled regions
  {
    id: 'head_neck_front',
    name: 'Head/Neck (Front)',
    description: 'Head and neck area - front view',
    coordinates: { x: 200, y: 15, width: 100, height: 95 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: 'chest',
    name: 'Chest',
    description: 'Chest area',
    coordinates: { x: 150, y: 110, width: 190, height: 100 },
    commonWoundTypes: ['surgical_wound', 'traumatic_wound']
  },
  {
    id: 'arms_front_left',
    name: 'Left Arm (Front)',
    description: 'Left arm - front view',
    coordinates: { x: 340, y: 110, width: 100, height: 280 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: 'arms_front_right',
    name: 'Right Arm (Front)',
    description: 'Right arm - front view',
    coordinates: { x: 50, y: 110, width: 100, height: 280 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: 'hand_palm_left',
    name: 'Left Palm',
    description: 'Left palm - front view',
    coordinates: { x: 385, y: 390, width: 80, height: 85 },
    commonWoundTypes: ['traumatic_wound', 'burns']
  },
  {
    id: 'hand_palm_right',
    name: 'Right Palm',
    description: 'Right palm - front view',
    coordinates: { x: 25, y: 390, width: 80, height: 85 },
    commonWoundTypes: ['traumatic_wound', 'burns']
  }
];

// Back body regions matching the layout shown in the user's image
export const backBodyRegions: BodyRegion[] = [
  // Numbered regions from the diagram
  {
    id: 'lower_back',
    name: 'Lower Back',
    description: 'Lower back region',
    coordinates: { x: 155, y: 240, width: 175, height: 100 },
    commonWoundTypes: ['pressure_injury', 'surgical_wound']
  },
  {
    id: '1',
    name: 'Buttox',
    description: 'Buttox/gluteal region',
    coordinates: { x: 155, y: 340, width: 175, height: 100 },
    commonWoundTypes: ['pressure_injury', 'surgical_wound']
  },
  {
    id: '2L',
    name: 'Left Upper Leg (Back)',
    description: 'Left upper leg/thigh back',
    coordinates: { x: 150, y: 440, width: 80, height: 140 },
    commonWoundTypes: ['pressure_injury', 'traumatic_wound']
  },
  {
    id: '2R',
    name: 'Right Upper Leg (Back)',
    description: 'Right upper leg/thigh back',
    coordinates: { x: 240, y: 440, width: 80, height: 140 },
    commonWoundTypes: ['pressure_injury', 'traumatic_wound']
  },
  {
    id: '3L',
    name: 'Left Lower Leg (Back)',
    description: 'Left lower leg/calf back',
    coordinates: { x: 150, y: 580, width: 80, height: 140 },
    commonWoundTypes: ['venous_ulcer', 'traumatic_wound']
  },
  {
    id: '3R',
    name: 'Right Lower Leg (Back)',
    description: 'Right lower leg/calf back',
    coordinates: { x: 240, y: 580, width: 80, height: 140 },
    commonWoundTypes: ['venous_ulcer', 'traumatic_wound']
  },
  {
    id: '4L',
    name: 'Left Heel/Sole of Foot',
    description: 'Left heel/sole of foot',
    coordinates: { x: 150, y: 720, width: 80, height: 85 },
    commonWoundTypes: ['pressure_injury', 'diabetic_ulcer']
  },
  {
    id: '4R',
    name: 'Right Heel/Sole of Foot',
    description: 'Right heel/sole of foot',
    coordinates: { x: 240, y: 720, width: 80, height: 85 },
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
    coordinates: { x: 155, y: 110, width: 175, height: 130 },
    commonWoundTypes: ['pressure_injury', 'surgical_wound']
  },
  {
    id: 'arms_back_left',
    name: 'Left Arm (Back)',
    description: 'Left arm - back view',
    coordinates: { x: 55, y: 110, width: 100, height: 290 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: 'arms_back_right',
    name: 'Right Arm (Back)',
    description: 'Right arm - back view',
    coordinates: { x: 330, y: 110, width: 100, height: 290 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: 'hand_back_left',
    name: 'Left Back of Hand',
    description: 'Left back of hand - back view',
    coordinates: { x: 25, y: 400, width: 80, height: 90 },
    commonWoundTypes: ['traumatic_wound', 'burns']
  },
  {
    id: 'hand_back_right',
    name: 'Right Back of Hand',
    description: 'Right back of hand - back view',
    coordinates: { x: 375, y: 400, width: 80, height: 90 },
    commonWoundTypes: ['traumatic_wound', 'burns']
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