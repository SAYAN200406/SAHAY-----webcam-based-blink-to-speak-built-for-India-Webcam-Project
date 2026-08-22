function pollFPS() {
  fetch('/fps')
    .then(res => res.json())
    .then(data => {
      document.getElementById('fpsDisplay').textContent = data.fps;
    })
    .catch(() => {
      // Flask not reachable yet — leave display as is
    });
}

setInterval(pollFPS, 1000);