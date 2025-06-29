import { useRef } from "react";
import { CloudUpload, Image, X } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

interface ImageUploadSectionProps {
  selectedFiles: File[];
  onFilesSelect: (files: File[]) => void;
}

export default function ImageUploadSection({ selectedFiles, onFilesSelect }: ImageUploadSectionProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    const validFiles: File[] = [];
    
    for (const file of files) {
      // Validate file type and size
      if (!['image/jpeg', 'image/png'].includes(file.type)) {
        alert(`File "${file.name}" is not a valid image. Please select JPG or PNG files only.`);
        continue;
      }
      
      if (file.size > 10 * 1024 * 1024) {
        alert(`File "${file.name}" is too large. Images must be under 10MB.`);
        continue;
      }
      
      validFiles.push(file);
    }
    
    if (validFiles.length > 0) {
      onFilesSelect([...selectedFiles, ...validFiles]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    const validFiles: File[] = [];
    
    for (const file of files) {
      if (!['image/jpeg', 'image/png'].includes(file.type)) {
        alert(`File "${file.name}" is not a valid image. Please select JPG or PNG files only.`);
        continue;
      }
      
      if (file.size > 10 * 1024 * 1024) {
        alert(`File "${file.name}" is too large. Images must be under 10MB.`);
        continue;
      }
      
      validFiles.push(file);
    }
    
    if (validFiles.length > 0) {
      onFilesSelect([...selectedFiles, ...validFiles]);
    }
  };

  const removeFile = (indexToRemove: number) => {
    const newFiles = selectedFiles.filter((_, index) => index !== indexToRemove);
    onFilesSelect(newFiles);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  return (
    <Card className="mb-6">
      <CardContent className="p-6">
        <div className="flex items-center mb-4">
          <CloudUpload className="text-medical-blue text-lg mr-2" />
          <h2 className="text-lg font-semibold text-gray-900">Upload Wound Image</h2>
        </div>
        
        {/* Image Upload Area */}
        <div 
          className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-medical-blue transition-colors cursor-pointer"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept=".jpg,.jpeg,.png"
            multiple
            onChange={handleFileChange}
          />
          <CloudUpload className="text-4xl text-gray-400 mb-4 mx-auto" />
          <p className="text-gray-600 mb-2">Drag and drop your wound images here</p>
          <p className="text-sm text-gray-500 mb-4">or click to browse (multiple files supported)</p>
          <button className="bg-medical-blue text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium">
            Choose Files
          </button>
          <p className="text-xs text-gray-400 mt-2">JPG, PNG up to 10MB each</p>
        </div>

        {/* File Preview Area */}
        {selectedFiles.length > 0 && (
          <div className="mt-4 space-y-2">
            <h4 className="text-sm font-medium text-gray-700">Selected Images ({selectedFiles.length})</h4>
            {selectedFiles.map((file, index) => (
              <div key={index} className="flex items-center justify-between bg-gray-50 p-3 rounded-lg">
                <div className="flex items-center">
                  <Image className="text-medical-blue mr-2" />
                  <span className="text-sm text-gray-700">{file.name}</span>
                  <span className="text-xs text-gray-500 ml-2">({formatFileSize(file.size)})</span>
                </div>
                <button 
                  onClick={() => removeFile(index)}
                  className="text-red-500 hover:text-red-700"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
