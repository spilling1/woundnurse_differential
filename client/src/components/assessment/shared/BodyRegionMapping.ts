export interface BodyRegion {
  id: string;
  name: string;
  displayName: string;
  coordinates: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  image: 'front' | 'back';
}

// Body region mapping data based on the provided labels
export const bodyRegions: BodyRegion[] = [
  // Front body regions
  { id: '272', name: 'Front Top of Head', displayName: 'Top of Head', coordinates: { x: 313, y: 10, width: 20, height: 15 }, image: 'front' },
  { id: '273', name: 'Right ForeHead', displayName: 'Right Forehead', coordinates: { x: 305, y: 25, width: 15, height: 10 }, image: 'front' },
  { id: '274', name: 'Left ForeHead', displayName: 'Left Forehead', coordinates: { x: 325, y: 25, width: 15, height: 10 }, image: 'front' },
  { id: '275', name: 'Right Eye', displayName: 'Right Eye', coordinates: { x: 300, y: 35, width: 10, height: 8 }, image: 'front' },
  { id: '276', name: 'Left Eye', displayName: 'Left Eye', coordinates: { x: 335, y: 35, width: 10, height: 8 }, image: 'front' },
  { id: '277', name: 'Right Face', displayName: 'Right Face', coordinates: { x: 295, y: 45, width: 15, height: 20 }, image: 'front' },
  { id: '278', name: 'Left Face', displayName: 'Left Face', coordinates: { x: 335, y: 45, width: 15, height: 20 }, image: 'front' },
  { id: '279', name: 'Mouth', displayName: 'Mouth', coordinates: { x: 318, y: 50, width: 10, height: 8 }, image: 'front' },
  { id: '280', name: 'Right Jaw', displayName: 'Right Jaw', coordinates: { x: 300, y: 60, width: 12, height: 10 }, image: 'front' },
  { id: '281', name: 'Left Jaw', displayName: 'Left Jaw', coordinates: { x: 333, y: 60, width: 12, height: 10 }, image: 'front' },
  { id: '282', name: 'Front Right Shoulder', displayName: 'Right Shoulder', coordinates: { x: 280, y: 75, width: 20, height: 15 }, image: 'front' },
  { id: '283', name: 'Throat', displayName: 'Throat', coordinates: { x: 318, y: 70, width: 10, height: 12 }, image: 'front' },
  { id: '284', name: 'Front Left Shoulder', displayName: 'Left Shoulder', coordinates: { x: 345, y: 75, width: 20, height: 15 }, image: 'front' },
  { id: '285', name: 'Front Right Shoulder Joint', displayName: 'Right Shoulder Joint', coordinates: { x: 275, y: 90, width: 15, height: 12 }, image: 'front' },
  { id: '286', name: 'Right Chest', displayName: 'Right Chest', coordinates: { x: 300, y: 100, width: 20, height: 25 }, image: 'front' },
  { id: '287', name: 'Left Chest', displayName: 'Left Chest', coordinates: { x: 325, y: 100, width: 20, height: 25 }, image: 'front' },
  { id: '288', name: 'Front Left Shoulder Joint', displayName: 'Left Shoulder Joint', coordinates: { x: 355, y: 90, width: 15, height: 12 }, image: 'front' },
  { id: '289', name: 'Front Right Arm Over', displayName: 'Right Upper Arm', coordinates: { x: 255, y: 110, width: 20, height: 30 }, image: 'front' },
  { id: '290', name: 'Front Left Arm Over', displayName: 'Left Upper Arm', coordinates: { x: 370, y: 110, width: 20, height: 30 }, image: 'front' },
  { id: '291', name: 'Right Ribs', displayName: 'Right Ribs', coordinates: { x: 295, y: 130, width: 25, height: 20 }, image: 'front' },
  { id: '292', name: 'Left Ribs', displayName: 'Left Ribs', coordinates: { x: 325, y: 130, width: 25, height: 20 }, image: 'front' },
  { id: '293', name: 'Front Right Elbows', displayName: 'Right Elbow', coordinates: { x: 250, y: 145, width: 15, height: 12 }, image: 'front' },
  { id: '294', name: 'Right Upper Abdomen', displayName: 'Right Upper Abdomen', coordinates: { x: 295, y: 155, width: 25, height: 20 }, image: 'front' },
  { id: '295', name: 'Center Upper Abdomen', displayName: 'Center Upper Abdomen', coordinates: { x: 315, y: 155, width: 15, height: 20 }, image: 'front' },
  { id: '296', name: 'Left Upper Abdomen', displayName: 'Left Upper Abdomen', coordinates: { x: 325, y: 155, width: 25, height: 20 }, image: 'front' },
  { id: '297', name: 'Front Left Elbow', displayName: 'Left Elbow', coordinates: { x: 380, y: 145, width: 15, height: 12 }, image: 'front' },
  { id: '298', name: 'Front Right Arm Under', displayName: 'Right Forearm', coordinates: { x: 245, y: 160, width: 20, height: 30 }, image: 'front' },
  { id: '299', name: 'Right Lower Abdomen', displayName: 'Right Lower Abdomen', coordinates: { x: 295, y: 180, width: 25, height: 20 }, image: 'front' },
  { id: '300', name: 'Center Lower Abdomen', displayName: 'Center Lower Abdomen', coordinates: { x: 315, y: 180, width: 15, height: 20 }, image: 'front' },
  { id: '301', name: 'Left Lower Abdomen', displayName: 'Left Lower Abdomen', coordinates: { x: 325, y: 180, width: 25, height: 20 }, image: 'front' },
  { id: '302', name: 'Front Left Arm Under', displayName: 'Left Forearm', coordinates: { x: 380, y: 160, width: 20, height: 30 }, image: 'front' },
  { id: '303', name: 'Front Right Wrist', displayName: 'Right Wrist', coordinates: { x: 240, y: 195, width: 12, height: 8 }, image: 'front' },
  { id: '304', name: 'Right Hip', displayName: 'Right Hip', coordinates: { x: 290, y: 205, width: 20, height: 15 }, image: 'front' },
  { id: '305', name: 'Genitalia', displayName: 'Genitalia', coordinates: { x: 318, y: 205, width: 10, height: 15 }, image: 'front' },
  { id: '306', name: 'Left Hip', displayName: 'Left Hip', coordinates: { x: 335, y: 205, width: 20, height: 15 }, image: 'front' },
  { id: '307', name: 'Front Left Wrist', displayName: 'Left Wrist', coordinates: { x: 393, y: 195, width: 12, height: 8 }, image: 'front' },
  { id: '308', name: 'Front Right Hand', displayName: 'Right Hand', coordinates: { x: 235, y: 205, width: 15, height: 20 }, image: 'front' },
  { id: '309', name: 'Right Upper Thigh', displayName: 'Right Upper Thigh', coordinates: { x: 290, y: 225, width: 20, height: 30 }, image: 'front' },
  { id: '310', name: 'Left Upper Thigh', displayName: 'Left Upper Thigh', coordinates: { x: 335, y: 225, width: 20, height: 30 }, image: 'front' },
  { id: '311', name: 'Front Left Hand', displayName: 'Left Hand', coordinates: { x: 395, y: 205, width: 15, height: 20 }, image: 'front' },
  { id: '312', name: 'Right Lower Thigh', displayName: 'Right Lower Thigh', coordinates: { x: 290, y: 260, width: 20, height: 30 }, image: 'front' },
  { id: '313', name: 'Left Lower Thigh', displayName: 'Left Lower Thigh', coordinates: { x: 335, y: 260, width: 20, height: 30 }, image: 'front' },
  { id: '314', name: 'Front Right Knee', displayName: 'Right Knee', coordinates: { x: 290, y: 295, width: 20, height: 15 }, image: 'front' },
  { id: '315', name: 'Front Left Knee', displayName: 'Left Knee', coordinates: { x: 335, y: 295, width: 20, height: 15 }, image: 'front' },
  { id: '316', name: 'Right Leg', displayName: 'Right Lower Leg', coordinates: { x: 290, y: 315, width: 20, height: 40 }, image: 'front' },
  { id: '317', name: 'Left Leg', displayName: 'Left Lower Leg', coordinates: { x: 335, y: 315, width: 20, height: 40 }, image: 'front' },
  { id: '318', name: 'Front Right Ankle', displayName: 'Right Ankle', coordinates: { x: 290, y: 360, width: 20, height: 12 }, image: 'front' },
  { id: '319', name: 'Front Left Ankle', displayName: 'Left Ankle', coordinates: { x: 335, y: 360, width: 20, height: 12 }, image: 'front' },
  { id: '320', name: 'Right Foot', displayName: 'Right Foot', coordinates: { x: 290, y: 375, width: 20, height: 25 }, image: 'front' },
  { id: '321', name: 'Left Foot', displayName: 'Left Foot', coordinates: { x: 335, y: 375, width: 20, height: 25 }, image: 'front' },

  // Back body regions
  { id: '235', name: 'Back Top of Head', displayName: 'Back of Head', coordinates: { x: 313, y: 10, width: 20, height: 15 }, image: 'back' },
  { id: '236', name: 'Left Back of Head', displayName: 'Left Back of Head', coordinates: { x: 295, y: 25, width: 15, height: 10 }, image: 'back' },
  { id: '237', name: 'Right Back of Head', displayName: 'Right Back of Head', coordinates: { x: 335, y: 25, width: 15, height: 10 }, image: 'back' },
  { id: '238', name: 'Back Neck', displayName: 'Back of Neck', coordinates: { x: 318, y: 40, width: 10, height: 15 }, image: 'back' },
  { id: '239', name: 'Back Left Shoulder', displayName: 'Left Shoulder', coordinates: { x: 295, y: 60, width: 20, height: 15 }, image: 'back' },
  { id: '240', name: 'Back Right Shoulder', displayName: 'Right Shoulder', coordinates: { x: 330, y: 60, width: 20, height: 15 }, image: 'back' },
  { id: '241', name: 'Back Left Shoulder Joint', displayName: 'Left Shoulder Joint', coordinates: { x: 275, y: 75, width: 15, height: 12 }, image: 'back' },
  { id: '242', name: 'Back Right Shoulder Joint', displayName: 'Right Shoulder Joint', coordinates: { x: 355, y: 75, width: 15, height: 12 }, image: 'back' },
  { id: '243', name: 'Left Arm Over', displayName: 'Left Upper Arm', coordinates: { x: 255, y: 95, width: 20, height: 30 }, image: 'back' },
  { id: '244', name: 'Upper Left Back', displayName: 'Upper Left Back', coordinates: { x: 295, y: 85, width: 20, height: 25 }, image: 'back' },
  { id: '245', name: 'Upper Right Back', displayName: 'Upper Right Back', coordinates: { x: 330, y: 85, width: 20, height: 25 }, image: 'back' },
  { id: '246', name: 'Right Arm Over', displayName: 'Right Upper Arm', coordinates: { x: 370, y: 95, width: 20, height: 30 }, image: 'back' },
  { id: '247', name: 'Upper Middle Back', displayName: 'Upper Middle Back', coordinates: { x: 315, y: 110, width: 15, height: 20 }, image: 'back' },
  { id: '248', name: 'Lower Middle Back', displayName: 'Lower Middle Back', coordinates: { x: 315, y: 135, width: 15, height: 20 }, image: 'back' },
  { id: '249', name: 'Lower Left Back', displayName: 'Lower Left Back', coordinates: { x: 295, y: 135, width: 20, height: 25 }, image: 'back' },
  { id: '250', name: 'Lower Right Back', displayName: 'Lower Right Back', coordinates: { x: 330, y: 135, width: 20, height: 25 }, image: 'back' },
  { id: '251', name: 'Back Left Elbow', displayName: 'Left Elbow', coordinates: { x: 250, y: 130, width: 15, height: 12 }, image: 'back' },
  { id: '252', name: 'Back Right Elbow', displayName: 'Right Elbow', coordinates: { x: 380, y: 130, width: 15, height: 12 }, image: 'back' },
  { id: '253', name: 'Left Arm Under', displayName: 'Left Forearm', coordinates: { x: 245, y: 145, width: 20, height: 30 }, image: 'back' },
  { id: '254', name: 'Buttock', displayName: 'Buttocks', coordinates: { x: 310, y: 165, width: 25, height: 20 }, image: 'back' },
  { id: '255', name: 'Right Arm Under', displayName: 'Right Forearm', coordinates: { x: 380, y: 145, width: 20, height: 30 }, image: 'back' },
  { id: '256', name: 'Back Left Wrist', displayName: 'Left Wrist', coordinates: { x: 240, y: 180, width: 12, height: 8 }, image: 'back' },
  { id: '257', name: 'Back Right Wrist', displayName: 'Right Wrist', coordinates: { x: 393, y: 180, width: 12, height: 8 }, image: 'back' },
  { id: '258', name: 'Back Left Hand', displayName: 'Left Hand', coordinates: { x: 235, y: 190, width: 15, height: 20 }, image: 'back' },
  { id: '259', name: 'Back Left Upper Thigh', displayName: 'Left Upper Thigh', coordinates: { x: 295, y: 190, width: 20, height: 30 }, image: 'back' },
  { id: '260', name: 'Back Right Upper Thigh', displayName: 'Right Upper Thigh', coordinates: { x: 330, y: 190, width: 20, height: 30 }, image: 'back' },
  { id: '261', name: 'Back Right Hand', displayName: 'Right Hand', coordinates: { x: 395, y: 190, width: 15, height: 20 }, image: 'back' },
  { id: '262', name: 'Back Left Lower Thigh', displayName: 'Left Lower Thigh', coordinates: { x: 295, y: 225, width: 20, height: 30 }, image: 'back' },
  { id: '263', name: 'Back Right Lower Thigh', displayName: 'Right Lower Thigh', coordinates: { x: 330, y: 225, width: 20, height: 30 }, image: 'back' },
  { id: '264', name: 'Left Back Knee', displayName: 'Left Knee', coordinates: { x: 295, y: 260, width: 20, height: 15 }, image: 'back' },
  { id: '265', name: 'Right Back Knee', displayName: 'Right Knee', coordinates: { x: 330, y: 260, width: 20, height: 15 }, image: 'back' },
  { id: '266', name: 'Back Left Calf', displayName: 'Left Calf', coordinates: { x: 295, y: 280, width: 20, height: 40 }, image: 'back' },
  { id: '267', name: 'Back Right Calf', displayName: 'Right Calf', coordinates: { x: 330, y: 280, width: 20, height: 40 }, image: 'back' },
  { id: '268', name: 'Back Left Ankle', displayName: 'Left Ankle', coordinates: { x: 295, y: 325, width: 20, height: 12 }, image: 'back' },
  { id: '269', name: 'Back Right Ankle', displayName: 'Right Ankle', coordinates: { x: 330, y: 325, width: 20, height: 12 }, image: 'back' },
  { id: '270', name: 'Back Left Foot', displayName: 'Left Foot', coordinates: { x: 295, y: 340, width: 20, height: 25 }, image: 'back' },
  { id: '271', name: 'Back Right Foot', displayName: 'Right Foot', coordinates: { x: 330, y: 340, width: 20, height: 25 }, image: 'back' },
];

export const getBodyRegionById = (id: string): BodyRegion | undefined => {
  return bodyRegions.find(region => region.id === id);
};

export const getBodyRegionsByImage = (image: 'front' | 'back'): BodyRegion[] => {
  return bodyRegions.filter(region => region.image === image);
};