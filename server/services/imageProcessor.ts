export async function validateImage(file: Express.Multer.File): Promise<string> {
  if (!file) {
    throw new Error("No image file provided");
  }

  // Check file size (already handled by multer, but double-check)
  if (file.size > 10 * 1024 * 1024) {
    throw new Error("Image must be under 10MB");
  }

  // Check file type
  if (!['image/jpeg', 'image/png'].includes(file.mimetype)) {
    throw new Error("Image must be PNG or JPG format");
  }

  // Convert to base64
  const base64 = file.buffer.toString('base64');
  
  return base64;
}

export function getImageInfo(file: Express.Multer.File) {
  return {
    filename: file.originalname,
    size: file.size,
    type: file.mimetype,
    sizeFormatted: formatFileSize(file.size)
  };
}

function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
