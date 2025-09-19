import React, { useState } from "react";
import { Upload } from "lucide-react";

export default function ImageUploader() {
  const [image, setImage] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="flex flex-col items-center">
      <label className="cursor-pointer w-full flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-xl p-6 bg-gray-50 hover:bg-gray-100 transition">
        <Upload className="w-10 h-10 text-gray-500 mb-3" />
        <span className="text-gray-600 font-medium">
          Click to upload or drag & drop
        </span>
        <input
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleFileChange}
        />
      </label>

      {image && (
        <div className="mt-6 w-full">
          <h2 className="text-lg font-semibold text-gray-700 mb-3 text-center">
            Preview
          </h2>
          <img
            src={image}
            alt="Uploaded Preview"
            className="w-full h-auto rounded-xl shadow-lg border"
          />
        </div>
      )}
    </div>
  );
}
