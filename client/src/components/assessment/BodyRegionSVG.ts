/**
 * SVG Path Definitions for Body Region Outlines
 * These paths trace the actual anatomical shapes in the body diagrams
 */

export interface BodyRegionSVG {
  id: string;
  name: string;
  description: string;
  svgPath: string;
  commonWoundTypes: string[];
}

// Front body regions with SVG path outlines
export const frontBodyRegionsSVG: BodyRegionSVG[] = [
  {
    id: '272',
    name: 'Front Top of Head',
    description: 'Top of head area',
    svgPath: 'M 300 10 Q 330 5 360 10 Q 365 25 360 40 Q 330 45 300 40 Q 295 25 300 10 Z',
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '273',
    name: 'Right Forehead',
    description: 'Right side of forehead',
    svgPath: 'M 280 25 Q 310 20 295 40 Q 285 45 280 40 Q 275 32 280 25 Z',
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '274',
    name: 'Left Forehead',
    description: 'Left side of forehead',
    svgPath: 'M 325 25 Q 355 20 365 40 Q 360 45 350 40 Q 340 32 325 25 Z',
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '275',
    name: 'Right Eye',
    description: 'Right eye area',
    svgPath: 'M 280 35 Q 300 30 310 35 Q 305 45 285 40 Q 275 37 280 35 Z',
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '276',
    name: 'Left Eye',
    description: 'Left eye area',
    svgPath: 'M 335 35 Q 355 30 365 35 Q 360 45 340 40 Q 330 37 335 35 Z',
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '277',
    name: 'Right Face',
    description: 'Right side of face',
    svgPath: 'M 270 45 Q 300 40 310 55 Q 305 70 285 65 Q 265 60 270 45 Z',
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '278',
    name: 'Left Face',
    description: 'Left side of face',
    svgPath: 'M 335 45 Q 365 40 375 55 Q 370 70 350 65 Q 330 60 335 45 Z',
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '279',
    name: 'Mouth',
    description: 'Mouth area',
    svgPath: 'M 300 55 Q 325 50 350 55 Q 345 65 325 60 Q 300 65 300 55 Z',
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '280',
    name: 'Right Jaw',
    description: 'Right jaw area',
    svgPath: 'M 260 65 Q 290 60 300 75 Q 295 85 275 80 Q 255 75 260 65 Z',
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '281',
    name: 'Left Jaw',
    description: 'Left jaw area',
    svgPath: 'M 345 65 Q 375 60 385 75 Q 380 85 360 80 Q 340 75 345 65 Z',
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  
  // Chest and shoulders - using elliptical shapes for torso
  {
    id: '282',
    name: 'Front Right Shoulder',
    description: 'Right shoulder area',
    svgPath: 'M 240 85 Q 270 80 285 95 Q 280 115 260 110 Q 235 105 240 85 Z',
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '283',
    name: 'Throat',
    description: 'Throat area',
    svgPath: 'M 305 85 Q 335 80 340 95 Q 335 105 315 100 Q 300 95 305 85 Z',
    commonWoundTypes: ['surgical_wound', 'traumatic_wound']
  },
  {
    id: '284',
    name: 'Front Left Shoulder',
    description: 'Left shoulder area',
    svgPath: 'M 360 85 Q 390 80 405 95 Q 400 115 380 110 Q 355 105 360 85 Z',
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '286',
    name: 'Right Chest',
    description: 'Right chest area',
    svgPath: 'M 270 115 Q 300 110 315 130 Q 310 150 290 145 Q 265 140 270 115 Z',
    commonWoundTypes: ['surgical_wound', 'pressure_ulcer']
  },
  {
    id: '287',
    name: 'Left Chest',
    description: 'Left chest area',
    svgPath: 'M 330 115 Q 360 110 375 130 Q 370 150 350 145 Q 325 140 330 115 Z',
    commonWoundTypes: ['surgical_wound', 'pressure_ulcer']
  },
  
  // Arms - using elongated shapes
  {
    id: '289',
    name: 'Front Right Arm Over',
    description: 'Right upper arm',
    svgPath: 'M 230 135 Q 250 130 260 150 Q 255 175 240 170 Q 225 165 230 135 Z',
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '290',
    name: 'Front Left Arm Over',
    description: 'Left upper arm',
    svgPath: 'M 385 135 Q 405 130 415 150 Q 410 175 395 170 Q 380 165 385 135 Z',
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  
  // Torso areas
  {
    id: '291',
    name: 'Right Ribs',
    description: 'Right rib area',
    svgPath: 'M 265 145 Q 295 140 305 160 Q 300 180 280 175 Q 260 170 265 145 Z',
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '292',
    name: 'Left Ribs',
    description: 'Left rib area',
    svgPath: 'M 340 145 Q 370 140 380 160 Q 375 180 355 175 Q 335 170 340 145 Z',
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  
  // Abdomen - using wider elliptical shapes
  {
    id: '294',
    name: 'Right Upper Abdomen',
    description: 'Right upper abdomen',
    svgPath: 'M 265 175 Q 295 170 305 190 Q 300 210 280 205 Q 260 200 265 175 Z',
    commonWoundTypes: ['surgical_wound', 'pressure_ulcer']
  },
  {
    id: '295',
    name: 'Center Upper Abdomen',
    description: 'Center upper abdomen',
    svgPath: 'M 300 175 Q 330 170 340 190 Q 335 210 315 205 Q 295 200 300 175 Z',
    commonWoundTypes: ['surgical_wound', 'pressure_ulcer']
  },
  {
    id: '296',
    name: 'Left Upper Abdomen',
    description: 'Left upper abdomen',
    svgPath: 'M 340 175 Q 370 170 380 190 Q 375 210 355 205 Q 335 200 340 175 Z',
    commonWoundTypes: ['surgical_wound', 'pressure_ulcer']
  },
  
  // Lower abdomen and hips
  {
    id: '299',
    name: 'Right Lower Abdomen',
    description: 'Right lower abdomen',
    svgPath: 'M 265 205 Q 295 200 305 220 Q 300 240 280 235 Q 260 230 265 205 Z',
    commonWoundTypes: ['surgical_wound', 'pressure_ulcer']
  },
  {
    id: '300',
    name: 'Center Lower Abdomen',
    description: 'Center lower abdomen',
    svgPath: 'M 300 205 Q 330 200 340 220 Q 335 240 315 235 Q 295 230 300 205 Z',
    commonWoundTypes: ['surgical_wound', 'pressure_ulcer']
  },
  {
    id: '301',
    name: 'Left Lower Abdomen',
    description: 'Left lower abdomen',
    svgPath: 'M 340 205 Q 370 200 380 220 Q 375 240 355 235 Q 335 230 340 205 Z',
    commonWoundTypes: ['surgical_wound', 'pressure_ulcer']
  },
  
  // Legs - using elongated shapes for thighs and lower legs
  {
    id: '309',
    name: 'Right Upper Thigh',
    description: 'Right upper thigh',
    svgPath: 'M 270 265 Q 290 260 300 285 Q 295 310 280 305 Q 265 300 270 265 Z',
    commonWoundTypes: ['pressure_ulcer', 'venous_ulcer']
  },
  {
    id: '310',
    name: 'Left Upper Thigh',
    description: 'Left upper thigh',
    svgPath: 'M 345 265 Q 365 260 375 285 Q 370 310 355 305 Q 340 300 345 265 Z',
    commonWoundTypes: ['pressure_ulcer', 'venous_ulcer']
  },
  {
    id: '312',
    name: 'Right Lower Thigh',
    description: 'Right lower thigh',
    svgPath: 'M 270 305 Q 290 300 300 325 Q 295 350 280 345 Q 265 340 270 305 Z',
    commonWoundTypes: ['pressure_ulcer', 'venous_ulcer']
  },
  {
    id: '313',
    name: 'Left Lower Thigh',
    description: 'Left lower thigh',
    svgPath: 'M 345 305 Q 365 300 375 325 Q 370 350 355 345 Q 340 340 345 305 Z',
    commonWoundTypes: ['pressure_ulcer', 'venous_ulcer']
  },
  
  // Knees - using circular shapes
  {
    id: '314',
    name: 'Front Right Knee',
    description: 'Right knee',
    svgPath: 'M 270 345 Q 290 340 300 355 Q 295 370 280 365 Q 265 360 270 345 Z',
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '315',
    name: 'Front Left Knee',
    description: 'Left knee',
    svgPath: 'M 345 345 Q 365 340 375 355 Q 370 370 355 365 Q 340 360 345 345 Z',
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  
  // Lower legs
  {
    id: '316',
    name: 'Right Leg',
    description: 'Right lower leg',
    svgPath: 'M 270 375 Q 290 370 300 400 Q 295 425 280 420 Q 265 415 270 375 Z',
    commonWoundTypes: ['venous_ulcer', 'arterial_ulcer', 'diabetic_ulcer']
  },
  {
    id: '317',
    name: 'Left Leg',
    description: 'Left lower leg',
    svgPath: 'M 345 375 Q 365 370 375 400 Q 370 425 355 420 Q 340 415 345 375 Z',
    commonWoundTypes: ['venous_ulcer', 'arterial_ulcer', 'diabetic_ulcer']
  },
  
  // Feet - using foot-like shapes
  {
    id: '320',
    name: 'Right Foot',
    description: 'Right foot',
    svgPath: 'M 270 455 Q 290 450 305 465 Q 300 485 280 480 Q 265 475 270 455 Z',
    commonWoundTypes: ['diabetic_ulcer', 'pressure_ulcer']
  },
  {
    id: '321',
    name: 'Left Foot',
    description: 'Left foot',
    svgPath: 'M 345 455 Q 365 450 380 465 Q 375 485 355 480 Q 340 475 345 455 Z',
    commonWoundTypes: ['diabetic_ulcer', 'pressure_ulcer']
  }
];

// Back body regions with SVG path outlines
export const backBodyRegionsSVG: BodyRegionSVG[] = [
  {
    id: '235',
    name: 'Back Top of Head',
    description: 'Top back of head',
    svgPath: 'M 300 10 Q 330 5 360 10 Q 365 25 360 40 Q 330 45 300 40 Q 295 25 300 10 Z',
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '236',
    name: 'Left Back of Head',
    description: 'Left back of head',
    svgPath: 'M 280 25 Q 310 20 295 40 Q 285 45 280 40 Q 275 32 280 25 Z',
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '237',
    name: 'Right Back of Head',
    description: 'Right back of head',
    svgPath: 'M 325 25 Q 355 20 365 40 Q 360 45 350 40 Q 340 32 325 25 Z',
    commonWoundTypes: ['traumatic_wound', 'surgical_wound']
  },
  {
    id: '238',
    name: 'Back Neck',
    description: 'Back of neck',
    svgPath: 'M 305 55 Q 335 50 340 70 Q 335 85 315 80 Q 300 75 305 55 Z',
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  
  // Back and shoulders
  {
    id: '239',
    name: 'Back Left Shoulder',
    description: 'Left shoulder blade',
    svgPath: 'M 270 85 Q 300 80 310 105 Q 305 125 285 120 Q 265 115 270 85 Z',
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '240',
    name: 'Back Right Shoulder',
    description: 'Right shoulder blade',
    svgPath: 'M 335 85 Q 365 80 375 105 Q 370 125 350 120 Q 330 115 335 85 Z',
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  
  // Back regions
  {
    id: '244',
    name: 'Upper Left Back',
    description: 'Upper left back',
    svgPath: 'M 265 125 Q 295 120 305 145 Q 300 165 280 160 Q 260 155 265 125 Z',
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '245',
    name: 'Upper Right Back',
    description: 'Upper right back',
    svgPath: 'M 340 125 Q 370 120 380 145 Q 375 165 355 160 Q 335 155 340 125 Z',
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '247',
    name: 'Upper Middle Back',
    description: 'Upper middle back',
    svgPath: 'M 300 155 Q 330 150 340 175 Q 335 195 315 190 Q 295 185 300 155 Z',
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  {
    id: '248',
    name: 'Lower Middle Back',
    description: 'Lower middle back',
    svgPath: 'M 300 185 Q 330 180 340 205 Q 335 225 315 220 Q 295 215 300 185 Z',
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  
  // Buttocks
  {
    id: '254',
    name: 'Buttock',
    description: 'Buttock area',
    svgPath: 'M 300 225 Q 330 220 340 245 Q 335 265 315 260 Q 295 255 300 225 Z',
    commonWoundTypes: ['pressure_ulcer', 'surgical_wound']
  },
  
  // Back legs
  {
    id: '259',
    name: 'Back Left Upper Thigh',
    description: 'Left upper thigh',
    svgPath: 'M 270 265 Q 290 260 300 285 Q 295 310 280 305 Q 265 300 270 265 Z',
    commonWoundTypes: ['pressure_ulcer', 'venous_ulcer']
  },
  {
    id: '260',
    name: 'Back Right Upper Thigh',
    description: 'Right upper thigh',
    svgPath: 'M 345 265 Q 365 260 375 285 Q 370 310 355 305 Q 340 300 345 265 Z',
    commonWoundTypes: ['pressure_ulcer', 'venous_ulcer']
  },
  {
    id: '262',
    name: 'Back Left Lower Thigh',
    description: 'Left lower thigh',
    svgPath: 'M 270 305 Q 290 300 300 325 Q 295 350 280 345 Q 265 340 270 305 Z',
    commonWoundTypes: ['pressure_ulcer', 'venous_ulcer']
  },
  {
    id: '263',
    name: 'Back Right Lower Thigh',
    description: 'Right lower thigh',
    svgPath: 'M 345 305 Q 365 300 375 325 Q 370 350 355 345 Q 340 340 345 305 Z',
    commonWoundTypes: ['pressure_ulcer', 'venous_ulcer']
  },
  
  // Back knees
  {
    id: '264',
    name: 'Left Back Knee',
    description: 'Left knee',
    svgPath: 'M 270 345 Q 290 340 300 355 Q 295 370 280 365 Q 265 360 270 345 Z',
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  {
    id: '265',
    name: 'Right Back Knee',
    description: 'Right knee',
    svgPath: 'M 345 345 Q 365 340 375 355 Q 370 370 355 365 Q 340 360 345 345 Z',
    commonWoundTypes: ['pressure_ulcer', 'traumatic_wound']
  },
  
  // Back calves
  {
    id: '266',
    name: 'Back Left Calf',
    description: 'Left calf',
    svgPath: 'M 270 375 Q 290 370 300 400 Q 295 425 280 420 Q 265 415 270 375 Z',
    commonWoundTypes: ['venous_ulcer', 'arterial_ulcer', 'diabetic_ulcer']
  },
  {
    id: '267',
    name: 'Back Right Calf',
    description: 'Right calf',
    svgPath: 'M 345 375 Q 365 370 375 400 Q 370 425 355 420 Q 340 415 345 375 Z',
    commonWoundTypes: ['venous_ulcer', 'arterial_ulcer', 'diabetic_ulcer']
  },
  
  // Back feet
  {
    id: '270',
    name: 'Back Left Foot',
    description: 'Left foot',
    svgPath: 'M 270 455 Q 290 450 305 465 Q 300 485 280 480 Q 265 475 270 455 Z',
    commonWoundTypes: ['diabetic_ulcer', 'pressure_ulcer']
  },
  {
    id: '271',
    name: 'Back Right Foot',
    description: 'Right foot',
    svgPath: 'M 345 455 Q 365 450 380 465 Q 375 485 355 480 Q 340 475 345 455 Z',
    commonWoundTypes: ['diabetic_ulcer', 'pressure_ulcer']
  }
];

// Helper functions
export const getBodyRegionsSVGByImage = (view: 'front' | 'back'): BodyRegionSVG[] => {
  return view === 'front' ? frontBodyRegionsSVG : backBodyRegionsSVG;
};

export const getBodyRegionSVGByCoordinates = (x: number, y: number, view: 'front' | 'back'): BodyRegionSVG | null => {
  const regions = getBodyRegionsSVGByImage(view);
  
  // For SVG paths, we need to check if point is inside the path
  // This is a simplified approach - in a real implementation, you'd use path.isPointInPath()
  for (const region of regions) {
    // Simple bounding box check for now
    // In a real implementation, you'd parse the SVG path and check if point is inside
    const pathBounds = parseSVGPathBounds(region.svgPath);
    if (x >= pathBounds.x && x <= pathBounds.x + pathBounds.width &&
        y >= pathBounds.y && y <= pathBounds.y + pathBounds.height) {
      return region;
    }
  }
  
  return null;
};

// Helper function to parse SVG path bounds (simplified)
function parseSVGPathBounds(path: string): { x: number; y: number; width: number; height: number } {
  const matches = path.match(/M\s*(\d+)\s*(\d+).*?Q\s*(\d+)\s*(\d+)/);
  if (matches) {
    const x = parseInt(matches[1]);
    const y = parseInt(matches[2]);
    const endX = parseInt(matches[3]);
    const endY = parseInt(matches[4]);
    return {
      x: Math.min(x, endX) - 15,
      y: Math.min(y, endY) - 15,
      width: Math.abs(endX - x) + 30,
      height: Math.abs(endY - y) + 30
    };
  }
  return { x: 0, y: 0, width: 0, height: 0 };
}