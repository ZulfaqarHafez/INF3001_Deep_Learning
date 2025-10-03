// Import pipeline directly from Transformers.js
import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

class ObjectDetector {
  constructor() {
    //variables to store the detection model and result
    this.model = null;
    this.detections = [];

    //Store references to DOM elements
    this.elements = {};
    this.initDOM();//link DOM elements
    this.bindEvents();//set up UI interactions
    this.loadModel(); // to load the object detection model method

    window.addEventListener("resize", () => {
    if (this.detections.length > 0) {
      this.draw(this.detections); // redraw boxes with new size
    }
    });

    // this.loadClassificationModel(); // Line to add the claddification model
  }
  //grab DOM elements and stoer them in the this.elements for easy access
  initDOM() {
    this.elements = {
      uploadArea:       document.getElementById("uploadArea"),
      uploadContent:    document.getElementById("uploadContent"),
      imageContainer:   document.getElementById("imageContainer"),
      uploadedImage:    document.getElementById("uploadedImage"),
      detectionCanvas:  document.getElementById("detectionCanvas"),
      fileInput:        document.getElementById("fileInput"),
      exampleBtn:       document.getElementById("exampleBtn"),
      resetBtn:         document.getElementById("resetBtn"),
      progressContainer:document.getElementById("progressContainer"),
      progressFill:     document.getElementById("progressFill"),
      statusDot:        document.getElementById("statusDot"),
      statusText:       document.getElementById("statusText"),
      detectionCount:   document.getElementById("detectionCount"),
      detectionsList:   document.getElementById("detectionsList"),
    };
  }

  //attach all the event listeners (file-upload, darag and drop and the buttons)
  bindEvents() {
    //click the upload area opens the file picker
    this.elements.uploadArea.addEventListener("click", () => {
      if (!this.hasImage()) this.elements.fileInput.click();
    });
    // Handle file selection
    this.elements.fileInput.addEventListener("change", (e) => {
      if (e.target.files?.length) this.handleFile(e.target.files[0]);
    });
    // Handle drag & drop
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
    //example image button that linked to the jpg example
    this.elements.exampleBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      this.loadImage("public/street-scene-with-cars-and-people.jpg");
    });
    //reset button to clear the image and detection
    this.elements.resetBtn.addEventListener("click", () => this.resetImage());
  }
  //future method when using classification
  // async loadClassificationModel() {
  //   try {
  //     this.setStatus("Loading classification model...", "loading");
  //     this.classificationModel = await pipeline('image-classification', 'Xenova/mobilenet_v2');
  //     this.setStatus("Classification model ready.", "ready");
  //   } catch (e) {
  //     console.error(e);
  //     this.setStatus("Error loading classification model.", "error");
  //   }
  // }
  // async classifyImage(image) {
  //   if (!this.classificationModel) {
  //     this.setStatus("Classification model not loaded.", "error");
  //     return;
  //   }
  //   this.setStatus("Classifying image...", "loading");
  //   const output = await this.classificationModel(image.src);
  //   this.setStatus("Image classified.", "ready");
  //   console.log(output);
  // }
  //load the Hugging Face DETR model for object detection
  async loadModel() {
    try {
      this.setStatus("Loading AI model…", "loading");
      this.showProgress(20);

      // this.model = await pipeline("object-detection", "Xenova/detr-resnet-101", {
      //   quantized: false,
      // });
      

      this.showProgress(100);
      this.setStatus("Ready", "ready");
      setTimeout(() => this.hideProgress(), 500);
    } catch (err) {
      console.error(err);
      this.setStatus("Error loading model", "error");
      this.hideProgress();
    }
  }
  async classifyImage(file) {
            this.setStatus("Sending to local model…", "loading");

            const formData = new FormData();
            formData.append("file", file);

            const res = await fetch("http://127.0.0.1:8000/predict", {
              method: "POST",
              body: formData,
            });

            const data = await res.json();
            console.log("Local model output:", data);

            this.setStatus("Prediction ready", "ready");
            this.renderDetectionsList(data.predictions);
          }
  //Handle file uploads from <input type='file'>
  // handleFile(file) {
  //   if (!file || !file.type.startsWith("image/")) {
  //     alert("Please select a valid image file.");
  //     return;
  //   }
  //   const reader = new FileReader();
  //   reader.onload = (e) => this.loadImage(e.target.result);
  //   reader.readAsDataURL(file);
  // }
  handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    alert("Please select a valid image file.");
    return;
  }
  this.classifyImage(file);
  }

  // Load an image into the preview <img>
  loadImage(src) {
    this.elements.uploadedImage.src = src;
    this.elements.uploadedImage.onload = () => {
      this.showImage();
      this.detect();
    };
  }
  //show the uploaded image then it will hide the upload placeholder
  showImage() {
    this.elements.uploadContent.style.display = "none";
    this.elements.imageContainer.style.display = "block";
    this.elements.uploadArea.classList.add("has-image");

    const img = this.elements.uploadedImage;
    const canvas = this.elements.detectionCanvas;
    canvas.width  = img.naturalWidth;
    canvas.height = img.naturalHeight;
    //canvas size is set dynamically during drawing, can improce later on
    canvas.style.width  = img.offsetWidth + "px";
    canvas.style.height = img.offsetHeight + "px";
  }
  //reset everything which is the UI and also the state
  resetImage() {
    this.elements.uploadContent.style.display = "flex";
    this.elements.imageContainer.style.display = "none";
    this.elements.uploadArea.classList.remove("has-image");
    this.elements.fileInput.value = "";
    this.detections = [];
    this.renderDetectionsList();
    this.clearCanvas();
    this.setStatus(this.model ? "Ready" : "Loading model…", this.model ? "ready" : "loading");
  }
  //check to see if an image is displayed
  hasImage() {
    return this.elements.imageContainer.style.display !== "none";
  }
  //run the object detection pipeline on the uploaded image
async detect() {
    if (!this.model) {
      this.setStatus("Model not ready yet", "error");
      return;
    }

    this.setStatus("Detecting objects…", "loading");
    this.showProgress(10);
    this.detections = [];
    this.renderDetectionsList();

    try {
      const imgEl = this.elements.uploadedImage;

      // The key change is here: pass the image's src attribute to the model
      const results = await this.model(imgEl.src, {
        threshold: 0.6,
        topk: 50,
      });


      this.showProgress(100);
      console.log("Raw results:", results); // Debug
      this.detections = results;
      this.draw(results);
      this.renderDetectionsList();
      this.setStatus(`Detected ${results.length} objects`, "ready");
      setTimeout(() => this.hideProgress(), 600);
    } catch (err) {
      console.error("Detection error:", err);
      this.setStatus("Detection failed", "error");
      this.hideProgress();
    }
  }


  //Draw bounding boxes and labels on the detection canvas
  draw(detections) {
  const canvas = this.elements.detectionCanvas;
  const ctx = canvas.getContext("2d");
  const img = this.elements.uploadedImage;

  // Set canvas pixel dimensions to match displayed size
  canvas.width = img.offsetWidth;
  canvas.height = img.offsetHeight;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Scale factor between detection (natural size) and displayed size
  const scaleX = img.offsetWidth / img.naturalWidth;
  const scaleY = img.offsetHeight / img.naturalHeight;

  const colors = ["#ff6b6b","#4ecdc4","#45b7d1","#96ceb4","#ffeaa7","#dda0dd","#98d8c8"];

  detections.forEach((d, i) => {
    const { box, label, score } = d;
    const color = colors[i % colors.length];

    const xmin = box.xmin * scaleX;
    const ymin = box.ymin * scaleY;
    const xmax = box.xmax * scaleX;
    const ymax = box.ymax * scaleY;

    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

    const text = `${label} (${Math.round(score * 100)}%)`;
    ctx.font = "14px Arial";
    const width = ctx.measureText(text).width + 8;
    ctx.fillStyle = color;
    ctx.fillRect(xmin, Math.max(0, ymin - 22), width, 22);

    ctx.fillStyle = "#fff";
    ctx.fillText(text, xmin + 4, Math.max(12, ymin - 6));
  });
}

// Render the detection list on the right-hand side panel
  // renderDetectionsList() {
  //   const list = this.elements.detectionsList;
  //   list.innerHTML = "";
  //   const count = this.detections.length;
  //   this.elements.detectionCount.textContent =
  //     count ? `${count} object${count !== 1 ? "s" : ""} detected` : "";

  //   if (!count) return;

  //   const sorted = [...this.detections].sort((a, b) => b.score - a.score);
  //   const colors = ["#ff6b6b","#4ecdc4","#45b7d1","#96ceb4","#ffeaa7","#dda0dd","#98d8c8"];

  //   sorted.forEach((d) => {
  //     const item = document.createElement("div");
  //     item.className = "detection-item";
  //     const idx = this.detections.indexOf(d);
  //     const color = colors[idx % colors.length];

  //     item.innerHTML = `
  //       <div class="detection-info">
  //         <div class="detection-color" style="background-color:${color};"></div>
  //         <span class="detection-label">${d.label}</span>
  //       </div>
  //       <span class="detection-confidence">${Math.round(d.score * 100)}% confidence</span>
  //     `;
  //     list.appendChild(item);
  //   });
  // }
  renderDetectionsList(predictions) {
  const list = this.elements.detectionsList;
  list.innerHTML = "";
  if (!predictions) return;

  predictions.sort((a, b) => b.probability - a.probability);

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
    this.elements.progressFill.style.width = `${Math.min(100, Math.max(0, pct))}%`;
  }

  hideProgress() {
    this.elements.progressContainer.style.display = "none";
    this.elements.progressFill.style.width = "0%";
  }

  clearCanvas() {
    const ctx = this.elements.detectionCanvas.getContext("2d");
    ctx.clearRect(0, 0, this.elements.detectionCanvas.width, this.elements.detectionCanvas.height);
  }
}

document.addEventListener("DOMContentLoaded", () => new ObjectDetector());
