function showInfo(message) {
  document.getElementById('info').textContent = '[INFO] ' + message;
}

async function upload() {
    const input = document.getElementById('cameraInput');
    const file = input.files[0];
    if (!file) {
      showInfo("Aucun fichier sélectionné !");
      return;
    }
    showInfo("Envoi de l'image en cours...");

    const formData = new FormData();
    formData.append('image', file);

    const response = await fetch('/process', {
      method: 'POST',
      body: formData
    });

    if (response.ok) {
      const blob = await response.blob();
      document.getElementById('result').src = URL.createObjectURL(blob);
      showInfo("Image traitée et reçue !");
    } else {
      showInfo("Erreur lors du traitement de l'image.");
    }
}

document.getElementById('cameraInput').addEventListener('change', function() {
  if (this.files.length > 0) {
    showInfo("Fichier sélectionné : " + this.files[0].name);
  }
});

// --- Selfie functionality ---
const selfieBtn = document.getElementById('selfieBtn');
const webcamContainer = document.getElementById('webcamContainer');
const webcam = document.getElementById('webcam');
const captureBtn = document.getElementById('captureBtn');
const selfieCanvas = document.getElementById('selfieCanvas');
const resultImg = document.getElementById('result');
let stream = null;

selfieBtn.onclick = async function() {
  webcamContainer.style.display = 'block';
  showInfo("Webcam activée. Cliquez sur 'Capturer' pour prendre la photo.");
  // Start webcam
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcam.srcObject = stream;
  } catch (err) {
    showInfo("Impossible d'accéder à la webcam : " + err);
  }
};

captureBtn.onclick = function() {
  // Draw the current frame from the video to the canvas
  selfieCanvas.getContext('2d').drawImage(webcam, 0, 0, selfieCanvas.width, selfieCanvas.height);
  // Stop the webcam
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }
  webcamContainer.style.display = 'none';
  // Show the captured image in the result <img>
  resultImg.src = selfieCanvas.toDataURL('image/png');
  showInfo("Selfie capturé et affiché !");
  // Optionally: you could also upload the selfie here
};