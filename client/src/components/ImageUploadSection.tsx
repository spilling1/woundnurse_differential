import { useRef } from "react";
import { CloudUpload, Image, X } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

interface ImageUploadSectionProps {
  selectedFile: File | null;
  onFileSelect: (file: File | null) => void;
}

export default function ImageUploadSection({ selectedFile, onFileSelect }: ImageUploadSectionProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Validate file type and size
      if (!['image/jpeg', 'image/png'].includes(file.type)) {
        alert('Please select a JPG or PNG image file.');
        return;
      }
      
      if (file.size > 10 * 1024 * 1024) {
        alert('Image size must be under 10MB.');
        return;
      }
      
      onFileSelect(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      if (!['image/jpeg', 'image/png'].includes(file.type)) {
        alert('Please select a JPG or PNG image file.');
        return;
      }
      
      if (file.size > 10 * 1024 * 1024) {
        alert('Image size must be under 10MB.');
        return;
      }
      
      onFileSelect(file);
    }
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
            onChange={handleFileChange}
          />
          <CloudUpload className="text-4xl text-gray-400 mb-4 mx-auto" />
          <p className="text-gray-600 mb-2">Drag and drop your wound image here</p>
          <p className="text-sm text-gray-500 mb-4">or click to browse</p>
          <button className="bg-medical-blue text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium">
            Choose File
          </button>
          <p className="text-xs text-gray-400 mt-2">JPG, PNG up to 10MB</p>
        </div>

        {/* File Preview Area */}
        {selectedFile && (
          <div className="mt-4">
            <div className="flex items-center justify-between bg-gray-50 p-3 rounded-lg">
              <div className="flex items-center">
                <Image className="text-medical-blue mr-2" />
                <span className="text-sm text-gray-700">{selectedFile.name}</span>
                <span className="text-xs text-gray-500 ml-2">({formatFileSize(selectedFile.size)})</span>
              </div>
              <button 
                onClick={() => onFileSelect(null)}
                className="text-red-500 hover:text-red-700"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
