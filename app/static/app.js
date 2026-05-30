const fileInput = document.querySelector("#file-input");
const chooseButton = document.querySelector("#choose-button");
const dropzone = document.querySelector("#dropzone");
const preview = document.querySelector("#preview");
const emptyPreview = document.querySelector("#empty-preview");
const statusDot = document.querySelector("#status-dot");
const statusText = document.querySelector("#status-text");
const predictionValue = document.querySelector("#prediction-value");
const normalScore = document.querySelector("#normal-score");
const sickScore = document.querySelector("#sick-score");
const normalBar = document.querySelector("#normal-bar");
const sickBar = document.querySelector("#sick-bar");
const message = document.querySelector("#message");

chooseButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) handleFile(file);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) handleFile(file);
});

async function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    setError("Please choose an image file.");
    return;
  }

  showPreview(file);
  setLoading();

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }

    setResult(data);
  } catch (error) {
    setError(error.message);
  }
}

function showPreview(file) {
  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
  emptyPreview.style.display = "none";
}

function setLoading() {
  statusDot.className = "status-dot";
  statusText.textContent = "Analyzing image";
  predictionValue.textContent = "--";
  normalScore.textContent = "--";
  sickScore.textContent = "--";
  normalBar.style.width = "0%";
  sickBar.style.width = "0%";
  message.textContent = "";
}

function setResult(data) {
  const normal = Math.max(0, Math.min(1, data.normal_probability));
  const sick = Math.max(0, Math.min(1, data.sick_probability));

  statusDot.className = "status-dot ready";
  statusText.textContent = "Analysis complete";
  predictionValue.textContent = data.prediction;
  normalScore.textContent = formatPercent(normal);
  sickScore.textContent = formatPercent(sick);
  normalBar.style.width = `${normal * 100}%`;
  sickBar.style.width = `${sick * 100}%`;
  message.textContent = "";
}

function setError(text) {
  statusDot.className = "status-dot error";
  statusText.textContent = "Needs attention";
  message.textContent = text;
}

function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}
