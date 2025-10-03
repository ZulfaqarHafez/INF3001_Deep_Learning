class ImageClassifier {
  constructor() {
    this.elements = {};
    this.initDOM();
    this.bindEvents();
  }

  initDOM() {
    this.elements = {
      uploadArea: document.getElementById("uploadArea"),
      uploadContent: document.getElementById("uploadContent"),
      imageContainer: document.getElementById("imageContainer"),
      uploadedImage: document.getElementById("uploadedImage"),
      fileInput: document.getElementById("fileInput"),
      exampleBtn: document.getElementById("exampleBtn"),
      resetBtn: document.getElementById("resetBtn"),
      progressContainer: document.getElementById("progressContainer"),
      progressFill: document.getElementById("progressFill"),
      statusDot: document.getElementById("statusDot"),
      statusText: document.getElementById("statusText"),
      detectionsList: document.getElementById("detectionsList"),
      topPrediction: document.getElementById("topPrediction"),
    };
  }

  bindEvents() {
    // Click upload area → file picker
    this.elements.uploadArea.addEventListener("click", () => {
      if (!this.hasImage()) this.elements.fileInput.click();
    });

    // File input change
    this.elements.fileInput.addEventListener("change", (e) => {
      if (e.target.files?.length) this.handleFile(e.target.files[0]);
    });

    // Example button → load local sample image
    this.elements.exampleBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      try {
        const res = await fetch("public/1.jpg");
        const blob = await res.blob();
        const file = new File([blob], "1.jpg", { type: blob.type });

        this.elements.uploadedImage.src = URL.createObjectURL(file);
        this.elements.uploadedImage.onload = () => {
          this.showImage();
          this.classifyImage(file);
        };
      } catch (err) {
        console.error("Error loading example image:", err);
      }
    });

    // Reset button
    this.elements.resetBtn.addEventListener("click", () => this.resetImage());

    // Drag & drop
    this.elements.uploadArea.addEventListener("dragover", (e) => {
      e.preventDefault();
      this.elements.uploadArea.classList.add("dragover");
    });
    this.elements.uploadArea.addEventListener("dragleave", () => {
      this.elements.uploadArea.classList.remove("dragover");
    });
    this.elements.uploadArea.addEventListener("drop", (e) => {
      e.preventDefault();
      this.elements.uploadArea.classList.remove("dragover");
      if (e.dataTransfer.files?.length) this.handleFile(e.dataTransfer.files[0]);
    });
  }

  hasImage() {
    return this.elements.imageContainer.style.display !== "none";
  }

  handleFile(file) {
    if (!file || !file.type.startsWith("image/")) {
      alert("Please select a valid image file.");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      this.elements.uploadedImage.src = e.target.result;
      this.elements.uploadedImage.onload = () => {
        this.showImage();
        this.classifyImage(file);
      };
    };
    reader.readAsDataURL(file);
  }

  showImage() {
    this.elements.uploadContent.style.display = "none";
    this.elements.imageContainer.style.display = "block";
    this.elements.uploadArea.classList.add("has-image");
  }

  resetImage() {
    this.elements.uploadContent.style.display = "flex";
    this.elements.imageContainer.style.display = "none";
    this.elements.uploadArea.classList.remove("has-image");
    this.elements.fileInput.value = "";
    this.elements.detectionsList.innerHTML = "";
    this.elements.topPrediction.textContent = "";
    this.setStatus("Waiting for image", "loading");
  }

  async classifyImage(file) {
    this.setStatus("Classifying image…", "loading");
    this.showProgress(20);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      this.showProgress(100);

      if (!data.predictions) {
        this.setStatus("No predictions returned", "error");
        return;
      }

      this.renderResults(data.predictions);
      this.setStatus("Classification complete", "ready");

    } catch (err) {
      console.error(err);
      this.setStatus("Error during classification", "error");
    }
  }

  renderResults(predictions) {
  const list = this.elements.detectionsList;
  list.innerHTML = "";

  if (!predictions || predictions.length === 0) {
    this.elements.topPrediction.textContent = "No predictions available.";
    this.elements.topPrediction.className = "";
    return;
  }

  // Sort by probability
  predictions.sort((a, b) => b.probability - a.probability);

  // Top prediction
  const top = predictions[0];
  this.elements.topPrediction.textContent =
    `Prediction: ${top.label} (${(top.probability * 100).toFixed(2)}%)`;

  // Reset classes then add one
  this.elements.topPrediction.className = "";
  if (top.label.toLowerCase() === "helmet") {
    this.elements.topPrediction.classList.add("helmet");
  } else {
    this.elements.topPrediction.classList.add("no_helmet");
  }

  // Show all predictions
  predictions.forEach((p) => {
    const item = document.createElement("div");
    item.className = "detection-item";
    item.innerHTML = `
      <div class="detection-info">
        <span class="detection-label">${p.label}</span>
      </div>
      <span class="detection-confidence">${(p.probability * 100).toFixed(2)}%</span>
    `;
    list.appendChild(item);
  });
}


  setStatus(text, type) {
    this.elements.statusText.textContent = text;
    this.elements.statusDot.className = `status-dot ${type}`;
  }

  showProgress(pct) {
    this.elements.progressContainer.style.display = "block";
    this.elements.progressFill.style.width = `${pct}%`;
    if (pct >= 100) {
      setTimeout(() => {
        this.elements.progressContainer.style.display = "none";
        this.elements.progressFill.style.width = "0%";
      }, 800);
    }
  }
}

document.addEventListener("DOMContentLoaded", () => new ImageClassifier());
