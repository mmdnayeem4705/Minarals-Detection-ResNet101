// Simple drag & drop + file name display

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const fileNameLabel = document.getElementById("file-name");
const uploadForm = document.getElementById("upload-form");

if (dropzone && fileInput && fileNameLabel) {
  dropzone.addEventListener("click", () => fileInput.click());

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      dropzone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      dropzone.classList.remove("drag-over");
    });
  });

  dropzone.addEventListener("drop", (event) => {
    const files = event.dataTransfer?.files;
    if (files && files.length > 0) {
      fileInput.files = files;
      fileNameLabel.textContent = files[0].name;
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      fileNameLabel.textContent = fileInput.files[0].name;
    } else {
      fileNameLabel.textContent = "";
    }
  });
}

if (uploadForm) {
  uploadForm.addEventListener("submit", () => {
    if (fileNameLabel) {
      fileNameLabel.textContent = "Uploading...";
    }
  });
}

