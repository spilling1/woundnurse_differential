/**
 * Body Region Mapping for Wound Assessment
 * Maps clickable coordinates to anatomical regions for front and back body diagrams
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

// Front body regions with coordinate mappings
export const frontBodyRegions: BodyRegion[] = [
  {
    id: 'head_front',
    name: 'Head/Face',
    description: 'Head, face, and neck area',
    coordinates: { x: 45, y: 5, width: 10, height: 15 },
    commonWoundTypes: ['surgical_wound', 'traumatic_wound']
  },
  {
    id: 'chest',
    name: 'Chest',
    description: 'Chest and upper torso',
    coordinates: { x: 35, y: 20, width: 30, height: 20 },
    commonWoundTypes: ['surgical_wound', 'pressure_ulcer']
  },
  {
    id: 'abdomen',
    name: 'Abdomen',
    description: 'Abdominal area',
    coordinates: { x: 35, y: 40, width: 30, height: 15 },
    commonWoundTypes: ['surgical_wound', 'pressure_ulcer']
  },
  {
    id: 'left_arm_front',
    name: 'Left Arm',
    description: 'Left arm and shoulder',
    coordinates: { x: 15, y: 20, width: 20, height: 35 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: 'right_arm_front',
    name: 'Right Arm',
    description: 'Right arm and shoulder',
    coordinates: { x: 65, y: 20, width: 20, height: 35 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: 'left_hand',
    name: 'Left Hand',
    description: 'Left hand and fingers',
    coordinates: { x: 10, y: 55, width: 10, height: 10 },
    commonWoundTypes: ['traumatic_wound', 'diabetic_ulcer']
  },
  {
    id: 'right_hand',
    name: 'Right Hand',
    description: 'Right hand and fingers',
    coordinates: { x: 80, y: 55, width: 10, height: 10 },
    commonWoundTypes: ['traumatic_wound', 'diabetic_ulcer']
  },
  {
    id: 'left_leg_front',
    name: 'Left Leg',
    description: 'Left thigh and shin',
    coordinates: { x: 35, y: 55, width: 15, height: 35 },
    commonWoundTypes: ['venous_ulcer', 'arterial_ulcer', 'diabetic_ulcer']
  },
  {
    id: 'right_leg_front',
    name: 'Right Leg',
    description: 'Right thigh and shin',
    coordinates: { x: 50, y: 55, width: 15, height: 35 },
    commonWoundTypes: ['venous_ulcer', 'arterial_ulcer', 'diabetic_ulcer']
  },
  {
    id: 'left_foot',
    name: 'Left Foot',
    description: 'Left foot and toes',
    coordinates: { x: 32, y: 90, width: 12, height: 8 },
    commonWoundTypes: ['diabetic_ulcer', 'pressure_ulcer']
  },
  {
    id: 'right_foot',
    name: 'Right Foot',
    description: 'Right foot and toes',
    coordinates: { x: 56, y: 90, width: 12, height: 8 },
    commonWoundTypes: ['diabetic_ulcer', 'pressure_ulcer']
  }
];

// Back body regions with coordinate mappings
export const backBodyRegions: BodyRegion[] = [
  {
    id: 'head_back',
    name: 'Back of Head',
    description: 'Back of head and neck',
    coordinates: { x: 45, y: 5, width: 10, height: 15 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: 'upper_back',
    name: 'Upper Back',
    description: 'Upper back and shoulders',
    coordinates: { x: 35, y: 20, width: 30, height: 20 },
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: 'lower_back',
    name: 'Lower Back',
    description: 'Lower back and lumbar area',
    coordinates: { x: 35, y: 40, width: 30, height: 15 },
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: 'sacrum',
    name: 'Sacrum/Tailbone',
    description: 'Sacral area and tailbone',
    coordinates: { x: 45, y: 55, width: 10, height: 8 },
    commonWoundTypes: ['pressure_ulcer']
  },
  {
    id: 'left_arm_back',
    name: 'Left Arm (Back)',
    description: 'Back of left arm',
    coordinates: { x: 15, y: 20, width: 20, height: 35 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: 'right_arm_back',
    name: 'Right Arm (Back)',
    description: 'Back of right arm',
    coordinates: { x: 65, y: 20, width: 20, height: 35 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: 'left_buttock',
    name: 'Left Buttock',
    description: 'Left buttock area',
    coordinates: { x: 35, y: 55, width: 15, height: 12 },
    commonWoundTypes: ['pressure_ulcer']
  },
  {
    id: 'right_buttock',
    name: 'Right Buttock',
    description: 'Right buttock area',
    coordinates: { x: 50, y: 55, width: 15, height: 12 },
    commonWoundTypes: ['pressure_ulcer']
  },
  {
    id: 'left_leg_back',
    name: 'Left Leg (Back)',
    description: 'Back of left thigh and calf',
    coordinates: { x: 35, y: 67, width: 15, height: 23 },
    commonWoundTypes: ['venous_ulcer', 'pressure_ulcer']
  },
  {
    id: 'right_leg_back',
    name: 'Right Leg (Back)',
    description: 'Back of right thigh and calf',
    coordinates: { x: 50, y: 67, width: 15, height: 23 },
    commonWoundTypes: ['venous_ulcer', 'pressure_ulcer']
  },
  {
    id: 'left_heel',
    name: 'Left Heel',
    description: 'Left heel area',
    coordinates: { x: 35, y: 90, width: 8, height: 8 },
    commonWoundTypes: ['pressure_ulcer', 'diabetic_ulcer']
  },
  {
    id: 'right_heel',
    name: 'Right Heel',
    description: 'Right heel area',
    coordinates: { x: 57, y: 90, width: 8, height: 8 },
    commonWoundTypes: ['pressure_ulcer', 'diabetic_ulcer']
  }
];

// Utility functions for body region mapping
export const getBodyRegionById = (id: string): BodyRegion | undefined => {
  return [...frontBodyRegions, ...backBodyRegions].find(region => region.id === id);
};

export const getBodyRegionByCoordinates = (x: number, y: number, view: 'front' | 'back'): BodyRegion | undefined => {
  const regions = view === 'front' ? frontBodyRegions : backBodyRegions;
  
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

export const getCommonWoundTypesForRegion = (regionId: string): string[] => {
  const region = getBodyRegionById(regionId);
  return region?.commonWoundTypes || [];
};