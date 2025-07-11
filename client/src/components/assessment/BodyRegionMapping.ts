/**
 * Body Region Mapping for Wound Assessment
 * Maps clickable coordinates to anatomical regions based on the numbered regions in body diagrams
 * Coordinates are positioned around the visible numbers in the images
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

// Front body regions with coordinate mappings around existing numbers (272-321)
export const frontBodyRegions: BodyRegion[] = [
  // Head region numbers
  {
    id: '272',
    name: 'Front Top of Head',
    description: 'Top of head area',
    coordinates: { x: 315, y: 15, width: 25, height: 20 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '273',
    name: 'Right Forehead',
    description: 'Right side of forehead',
    coordinates: { x: 295, y: 25, width: 25, height: 15 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '274',
    name: 'Left Forehead',
    description: 'Left side of forehead',
    coordinates: { x: 325, y: 25, width: 25, height: 15 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '275',
    name: 'Right Eye',
    description: 'Right eye area',
    coordinates: { x: 290, y: 35, width: 20, height: 15 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '276',
    name: 'Left Eye',
    description: 'Left eye area',
    coordinates: { x: 335, y: 35, width: 20, height: 15 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '277',
    name: 'Right Face',
    description: 'Right side of face',
    coordinates: { x: 285, y: 45, width: 25, height: 20 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '278',
    name: 'Left Face',
    description: 'Left side of face',
    coordinates: { x: 335, y: 45, width: 25, height: 20 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '279',
    name: 'Mouth',
    description: 'Mouth area',
    coordinates: { x: 305, y: 55, width: 20, height: 15 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '280',
    name: 'Right Jaw',
    description: 'Right jaw area',
    coordinates: { x: 275, y: 65, width: 25, height: 15 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '281',
    name: 'Left Jaw',
    description: 'Left jaw area',
    coordinates: { x: 345, y: 65, width: 25, height: 15 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  
  // Shoulder and chest region
  {
    id: '282',
    name: 'Front Right Shoulder',
    description: 'Right shoulder area',
    coordinates: { x: 255, y: 85, width: 25, height: 20 },
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '283',
    name: 'Throat',
    description: 'Throat area',
    coordinates: { x: 310, y: 85, width: 20, height: 15 },
    commonWoundTypes: ['surgical_wound', 'traumatic_wound']
  },
  {
    id: '284',
    name: 'Front Left Shoulder',
    description: 'Left shoulder area',
    coordinates: { x: 365, y: 85, width: 25, height: 20 },
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '285',
    name: 'Front Right Shoulder Joint',
    description: 'Right shoulder joint',
    coordinates: { x: 250, y: 105, width: 25, height: 20 },
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '286',
    name: 'Right Chest',
    description: 'Right chest area',
    coordinates: { x: 280, y: 115, width: 25, height: 25 },
    commonWoundTypes: ['surgical_wound', 'pressure_ulcer']
  },
  {
    id: '287',
    name: 'Left Chest',
    description: 'Left chest area',
    coordinates: { x: 340, y: 115, width: 25, height: 25 },
    commonWoundTypes: ['surgical_wound', 'pressure_ulcer']
  },
  {
    id: '288',
    name: 'Front Left Shoulder Joint',
    description: 'Left shoulder joint',
    coordinates: { x: 370, y: 105, width: 25, height: 20 },
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  
  // Arms
  {
    id: '289',
    name: 'Front Right Arm Over',
    description: 'Right upper arm',
    coordinates: { x: 240, y: 135, width: 25, height: 30 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '290',
    name: 'Front Left Arm Over',
    description: 'Left upper arm',
    coordinates: { x: 380, y: 135, width: 25, height: 30 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '291',
    name: 'Right Ribs',
    description: 'Right rib area',
    coordinates: { x: 275, y: 145, width: 25, height: 25 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '292',
    name: 'Left Ribs',
    description: 'Left rib area',
    coordinates: { x: 345, y: 145, width: 25, height: 25 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '293',
    name: 'Front Right Elbow',
    description: 'Right elbow',
    coordinates: { x: 235, y: 175, width: 25, height: 20 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '294',
    name: 'Right Upper Abdomen',
    description: 'Right upper abdomen',
    coordinates: { x: 275, y: 175, width: 25, height: 25 },
    commonWoundTypes: ['surgical_wound', 'pressure_ulcer']
  },
  {
    id: '295',
    name: 'Center Upper Abdomen',
    description: 'Center upper abdomen',
    coordinates: { x: 310, y: 175, width: 25, height: 25 },
    commonWoundTypes: ['surgical_wound', 'pressure_ulcer']
  },
  {
    id: '296',
    name: 'Left Upper Abdomen',
    description: 'Left upper abdomen',
    coordinates: { x: 345, y: 175, width: 25, height: 25 },
    commonWoundTypes: ['surgical_wound', 'pressure_ulcer']
  },
  {
    id: '297',
    name: 'Front Left Elbow',
    description: 'Left elbow',
    coordinates: { x: 385, y: 175, width: 25, height: 20 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '298',
    name: 'Front Right Arm Under',
    description: 'Right lower arm',
    coordinates: { x: 230, y: 205, width: 25, height: 30 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '299',
    name: 'Right Lower Abdomen',
    description: 'Right lower abdomen',
    coordinates: { x: 275, y: 205, width: 25, height: 25 },
    commonWoundTypes: ['surgical_wound', 'pressure_ulcer']
  },
  {
    id: '300',
    name: 'Center Lower Abdomen',
    description: 'Center lower abdomen',
    coordinates: { x: 310, y: 205, width: 25, height: 25 },
    commonWoundTypes: ['surgical_wound', 'pressure_ulcer']
  },
  {
    id: '301',
    name: 'Left Lower Abdomen',
    description: 'Left lower abdomen',
    coordinates: { x: 345, y: 205, width: 25, height: 25 },
    commonWoundTypes: ['surgical_wound', 'pressure_ulcer']
  },
  {
    id: '302',
    name: 'Front Left Arm Under',
    description: 'Left lower arm',
    coordinates: { x: 390, y: 205, width: 25, height: 30 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '303',
    name: 'Front Right Wrist',
    description: 'Right wrist',
    coordinates: { x: 225, y: 245, width: 25, height: 20 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '304',
    name: 'Right Hip',
    description: 'Right hip',
    coordinates: { x: 275, y: 235, width: 25, height: 25 },
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '305',
    name: 'Genitalia',
    description: 'Genital area',
    coordinates: { x: 310, y: 235, width: 25, height: 25 },
    commonWoundTypes: ['surgical_wound', 'traumatic_wound']
  },
  {
    id: '306',
    name: 'Left Hip',
    description: 'Left hip',
    coordinates: { x: 345, y: 235, width: 25, height: 25 },
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '307',
    name: 'Front Left Wrist',
    description: 'Left wrist',
    coordinates: { x: 395, y: 245, width: 25, height: 20 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '308',
    name: 'Front Right Hand',
    description: 'Right hand',
    coordinates: { x: 220, y: 275, width: 25, height: 25 },
    commonWoundTypes: ['diabetic_ulcer', 'traumatic_wound']
  },
  {
    id: '309',
    name: 'Right Upper Thigh',
    description: 'Right upper thigh',
    coordinates: { x: 275, y: 265, width: 25, height: 30 },
    commonWoundTypes: ['pressure_ulcer', 'venous_ulcer']
  },
  {
    id: '310',
    name: 'Left Upper Thigh',
    description: 'Left upper thigh',
    coordinates: { x: 345, y: 265, width: 25, height: 30 },
    commonWoundTypes: ['pressure_ulcer', 'venous_ulcer']
  },
  {
    id: '311',
    name: 'Front Left Hand',
    description: 'Left hand',
    coordinates: { x: 400, y: 275, width: 25, height: 25 },
    commonWoundTypes: ['diabetic_ulcer', 'traumatic_wound']
  },
  {
    id: '312',
    name: 'Right Lower Thigh',
    description: 'Right lower thigh',
    coordinates: { x: 275, y: 305, width: 25, height: 30 },
    commonWoundTypes: ['pressure_ulcer', 'venous_ulcer']
  },
  {
    id: '313',
    name: 'Left Lower Thigh',
    description: 'Left lower thigh',
    coordinates: { x: 345, y: 305, width: 25, height: 30 },
    commonWoundTypes: ['pressure_ulcer', 'venous_ulcer']
  },
  {
    id: '314',
    name: 'Front Right Knee',
    description: 'Right knee',
    coordinates: { x: 275, y: 345, width: 25, height: 25 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '315',
    name: 'Front Left Knee',
    description: 'Left knee',
    coordinates: { x: 345, y: 345, width: 25, height: 25 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '316',
    name: 'Right Leg',
    description: 'Right lower leg',
    coordinates: { x: 275, y: 375, width: 25, height: 40 },
    commonWoundTypes: ['venous_ulcer', 'arterial_ulcer', 'diabetic_ulcer']
  },
  {
    id: '317',
    name: 'Left Leg',
    description: 'Left lower leg',
    coordinates: { x: 345, y: 375, width: 25, height: 40 },
    commonWoundTypes: ['venous_ulcer', 'arterial_ulcer', 'diabetic_ulcer']
  },
  {
    id: '318',
    name: 'Front Right Ankle',
    description: 'Right ankle',
    coordinates: { x: 275, y: 425, width: 25, height: 20 },
    commonWoundTypes: ['pressure_ulcer', 'diabetic_ulcer']
  },
  {
    id: '319',
    name: 'Front Left Ankle',
    description: 'Left ankle',
    coordinates: { x: 345, y: 425, width: 25, height: 20 },
    commonWoundTypes: ['pressure_ulcer', 'diabetic_ulcer']
  },
  {
    id: '320',
    name: 'Right Foot',
    description: 'Right foot',
    coordinates: { x: 275, y: 455, width: 25, height: 30 },
    commonWoundTypes: ['diabetic_ulcer', 'pressure_ulcer']
  },
  {
    id: '321',
    name: 'Left Foot',
    description: 'Left foot',
    coordinates: { x: 345, y: 455, width: 25, height: 30 },
    commonWoundTypes: ['diabetic_ulcer', 'pressure_ulcer']
  }
];

// Back body regions with coordinate mappings around existing numbers (235-271)
export const backBodyRegions: BodyRegion[] = [
  {
    id: '235',
    name: 'Back Top of Head',
    description: 'Top back of head',
    coordinates: { x: 315, y: 15, width: 25, height: 20 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '236',
    name: 'Left Back of Head',
    description: 'Left back of head',
    coordinates: { x: 290, y: 25, width: 25, height: 20 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '237',
    name: 'Right Back of Head',
    description: 'Right back of head',
    coordinates: { x: 340, y: 25, width: 25, height: 20 },
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '238',
    name: 'Back Neck',
    description: 'Back of neck',
    coordinates: { x: 310, y: 55, width: 25, height: 20 },
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '239',
    name: 'Back Left Shoulder',
    description: 'Left shoulder blade',
    coordinates: { x: 280, y: 85, width: 25, height: 25 },
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '240',
    name: 'Back Right Shoulder',
    description: 'Right shoulder blade',
    coordinates: { x: 340, y: 85, width: 25, height: 25 },
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '241',
    name: 'Back Left Shoulder Joint',
    description: 'Left shoulder joint',
    coordinates: { x: 255, y: 105, width: 25, height: 20 },
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '242',
    name: 'Back Right Shoulder Joint',
    description: 'Right shoulder joint',
    coordinates: { x: 365, y: 105, width: 25, height: 20 },
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '243',
    name: 'Left Arm Over',
    description: 'Left upper arm',
    coordinates: { x: 240, y: 135, width: 25, height: 30 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '244',
    name: 'Upper Left Back',
    description: 'Upper left back',
    coordinates: { x: 275, y: 125, width: 25, height: 25 },
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '245',
    name: 'Upper Right Back',
    description: 'Upper right back',
    coordinates: { x: 345, y: 125, width: 25, height: 25 },
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '246',
    name: 'Right Arm Over',
    description: 'Right upper arm',
    coordinates: { x: 380, y: 135, width: 25, height: 30 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '247',
    name: 'Upper Middle Back',
    description: 'Upper middle back',
    coordinates: { x: 310, y: 155, width: 25, height: 25 },
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '248',
    name: 'Lower Middle Back',
    description: 'Lower middle back',
    coordinates: { x: 310, y: 185, width: 25, height: 25 },
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '249',
    name: 'Lower Left Back',
    description: 'Lower left back',
    coordinates: { x: 275, y: 185, width: 25, height: 25 },
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '250',
    name: 'Lower Right Back',
    description: 'Lower right back',
    coordinates: { x: 345, y: 185, width: 25, height: 25 },
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '251',
    name: 'Back Left Elbow',
    description: 'Left elbow',
    coordinates: { x: 235, y: 175, width: 25, height: 20 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '252',
    name: 'Back Right Elbow',
    description: 'Right elbow',
    coordinates: { x: 385, y: 175, width: 25, height: 20 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '253',
    name: 'Left Arm Under',
    description: 'Left lower arm',
    coordinates: { x: 230, y: 205, width: 25, height: 30 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '254',
    name: 'Buttock',
    description: 'Buttock area',
    coordinates: { x: 310, y: 225, width: 25, height: 25 },
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '255',
    name: 'Right Arm Under',
    description: 'Right lower arm',
    coordinates: { x: 390, y: 205, width: 25, height: 30 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '256',
    name: 'Back Left Wrist',
    description: 'Left wrist',
    coordinates: { x: 225, y: 245, width: 25, height: 20 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '257',
    name: 'Back Right Wrist',
    description: 'Right wrist',
    coordinates: { x: 395, y: 245, width: 25, height: 20 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '258',
    name: 'Back Left Hand',
    description: 'Left hand',
    coordinates: { x: 220, y: 275, width: 25, height: 25 },
    commonWoundTypes: ['diabetic_ulcer', 'traumatic_wound']
  },
  {
    id: '259',
    name: 'Back Left Upper Thigh',
    description: 'Left upper thigh',
    coordinates: { x: 275, y: 265, width: 25, height: 30 },
    commonWoundTypes: ['pressure_ulcer', 'venous_ulcer']
  },
  {
    id: '260',
    name: 'Back Right Upper Thigh',
    description: 'Right upper thigh',
    coordinates: { x: 345, y: 265, width: 25, height: 30 },
    commonWoundTypes: ['pressure_ulcer', 'venous_ulcer']
  },
  {
    id: '261',
    name: 'Back Right Hand',
    description: 'Right hand',
    coordinates: { x: 400, y: 275, width: 25, height: 25 },
    commonWoundTypes: ['diabetic_ulcer', 'traumatic_wound']
  },
  {
    id: '262',
    name: 'Back Left Lower Thigh',
    description: 'Left lower thigh',
    coordinates: { x: 275, y: 305, width: 25, height: 30 },
    commonWoundTypes: ['pressure_ulcer', 'venous_ulcer']
  },
  {
    id: '263',
    name: 'Back Right Lower Thigh',
    description: 'Right lower thigh',
    coordinates: { x: 345, y: 305, width: 25, height: 30 },
    commonWoundTypes: ['pressure_ulcer', 'venous_ulcer']
  },
  {
    id: '264',
    name: 'Left Back Knee',
    description: 'Left knee',
    coordinates: { x: 275, y: 345, width: 25, height: 25 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '265',
    name: 'Right Back Knee',
    description: 'Right knee',
    coordinates: { x: 345, y: 345, width: 25, height: 25 },
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '266',
    name: 'Back Left Calf',
    description: 'Left calf',
    coordinates: { x: 275, y: 375, width: 25, height: 40 },
    commonWoundTypes: ['venous_ulcer', 'arterial_ulcer', 'diabetic_ulcer']
  },
  {
    id: '267',
    name: 'Back Right Calf',
    description: 'Right calf',
    coordinates: { x: 345, y: 375, width: 25, height: 40 },
    commonWoundTypes: ['venous_ulcer', 'arterial_ulcer', 'diabetic_ulcer']
  },
  {
    id: '268',
    name: 'Back Left Ankle',
    description: 'Left ankle',
    coordinates: { x: 275, y: 425, width: 25, height: 20 },
    commonWoundTypes: ['pressure_ulcer', 'diabetic_ulcer']
  },
  {
    id: '269',
    name: 'Back Right Ankle',
    description: 'Right ankle',
    coordinates: { x: 345, y: 425, width: 25, height: 20 },
    commonWoundTypes: ['pressure_ulcer', 'diabetic_ulcer']
  },
  {
    id: '270',
    name: 'Back Left Foot',
    description: 'Left foot',
    coordinates: { x: 275, y: 455, width: 25, height: 30 },
    commonWoundTypes: ['diabetic_ulcer', 'pressure_ulcer']
  },
  {
    id: '271',
    name: 'Back Right Foot',
    description: 'Right foot',
    coordinates: { x: 345, y: 455, width: 25, height: 30 },
    commonWoundTypes: ['diabetic_ulcer', 'pressure_ulcer']
  }
];

// Helper function to get regions by image view
export const getBodyRegionsByImage = (view: 'front' | 'back'): BodyRegion[] => {
  return view === 'front' ? frontBodyRegions : backBodyRegions;
};

// Helper function to find region by coordinates
export const getBodyRegionByCoordinates = (x: number, y: number, view: 'front' | 'back'): BodyRegion | null => {
  const regions = getBodyRegionsByImage(view);
  
  for (const region of regions) {
    const { x: rx, y: ry, width, height } = region.coordinates;
    if (x >= rx && x <= rx + width && y >= ry && y <= ry + height) {
      return region;
    }
  }
  
  return null;
};