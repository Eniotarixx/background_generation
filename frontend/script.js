async function upload() {
    const input = document.getElementById('cameraInput');
    const file = input.files[0];
    const formData = new FormData();
    formData.append('image', file);

    const response = await fetch('/process', {
      method: 'POST',
      body: formData
    });

    const blob = await response.blob();
    document.getElementById('result').src = URL.createObjectURL(blob);
  }