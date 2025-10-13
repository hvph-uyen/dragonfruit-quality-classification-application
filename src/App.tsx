import React, { useState } from "react";
import "./index.css";

interface ImageItem {
  file: File;
  url: string;
  prediction: string;
}

const App: React.FC = () => {
  const [images, setImages] = useState<ImageItem[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const maxImages = 3;

// Upload từ file input (chọn hoặc chụp)
  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    const files = Array.from(e.target.files).slice(0, maxImages - images.length);
    const newImages = files.map((file) => ({
      file,
      url: URL.createObjectURL(file),
      prediction: "",
    }));
    setImages((prev) => [...prev, ...newImages]);
  };

// Kéo thả
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    if (!e.dataTransfer.files) return;
    const files = Array.from(e.dataTransfer.files).slice(0, maxImages - images.length);
    const newImages = files.map((file) => ({
      file,
      url: URL.createObjectURL(file),
      prediction: "",
    }));
    setImages((prev) => [...prev, ...newImages]);
  };

// Xóa ảnh
  const handleRemove = (index: number) => {
    setImages((prev) => prev.filter((_, i) => i !== index));
  };

  // Gọi API predict
  const handlePredict = async () => {
    const updatedImages = await Promise.all(
      images.map(async (img) => {
        const formData = new FormData();
        formData.append("file", img.file);
        try {
          const res = await fetch(`${window.location.origin.replace("5173", "8000")}/predict`, { // đổi IP laptop
            method: "POST",
            body: formData,
          });
          const data = await res.json();
          return { ...img, prediction: data.final_result };
        } catch (err) {
          console.error(err);
          return { ...img, prediction: "Lỗi server!" };
        }
      })
    );
    setImages(updatedImages);
  };

  const getFinalResult = () => {
    if (images.some((img) => img.prediction.toLowerCase() === "not_dragonfruit")) return "Not a Dragon Fruit";
    if (images.some((img) => img.prediction.toLowerCase() === "reject")) return "Reject";
    if (images.some((img) => img.prediction.toLowerCase() === "immature")) return "Immature";
    if (images.every((img) => img.prediction.toLowerCase() === "good")) return "Good";
    return "";
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🐉 DRAGONSCAN 🐉</h1>
      </header>

      <main className="main">
        {/* Upload box */}
        <div
          className={`upload-box ${dragOver ? "drag-over" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          {images.length === 0 && <p>Kéo thả ảnh hoặc chọn/chụp để tải lên</p>}
          <input
            type="file"
            accept="image/*"
            className="file-input"
            multiple
            onChange={handleUpload}
            disabled={images.length >= maxImages}
          />

          <div className="preview-container">
            {images.map((img, idx) => (
              <div key={idx} className="preview-item">
                <img src={img.url} alt={`Uploaded ${idx}`} className="preview" />
                <button className="remove-btn" onClick={() => handleRemove(idx)}>
                  ✖
                </button>
                {img.prediction && (
                  <div className={`img-result ${img.prediction.toLowerCase().replace(/\s/g, '-')}`}>
                    {img.prediction === "not_dragonfruit" ? "Not a Dragon Fruit" : img.prediction}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* mở camera */}
        <div className="camera-upload">
          <label className="camera-btn">
            📷 Chụp ảnh
            <input
              type="file"
              accept="image/*"
              capture="camera"
              onChange={handleUpload}
              style={{ display: "none" }}
            />
          </label>
        </div>

        {images.length > 0 && (
          <button className="predict-btn" onClick={handlePredict}>
            Kiểm định chất lượng
          </button>
        )}

        {images.some((img) => img.prediction) && (
          <div className={`final-result ${getFinalResult().toLowerCase()}`}>
            <h2>Kết quả: {getFinalResult()}</h2>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>© 2025 Dragonfruit Quality Classification - JOLIBEE</p>
      </footer>
    </div>
  );
};

export default App;
