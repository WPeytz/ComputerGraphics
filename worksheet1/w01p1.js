"use strict";
// Run after DOM is ready and surface any async errors
window.onload = () => { main().catch(err => showError(String(err))); };

async function main() {
  // 1) Feature detection with a helpful message
  if (!('gpu' in navigator) || !navigator.gpu) {
    showError(
      "WebGPU is not available (navigator.gpu is undefined).\n" +
      "Try one of these fixes:\n\n" +
      "• Open the site via http://localhost:8000 or http://127.0.0.1:8000 (avoid [::]).\n" +
      "• Use a recent Chrome/Edge (v113+).\n" +
      "• Ensure you're not opening the file from disk; use a local server.\n"
    );
    return;
  }

  const canvas = document.getElementById('my-canvas');
  if (!canvas) {
    showError("No <canvas id='my-canvas'> found in the DOM.");
    return;
  }

  // 2) Prefer sizing before configure (account for HiDPI)
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.floor((canvas.clientWidth || 512) * dpr);
  const h = Math.floor((canvas.clientHeight || 512) * dpr);
  canvas.width = w;
  canvas.height = h;

  // 3) Request adapter + device
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    showError("Failed to acquire a GPU adapter.");
    return;
  }

  const device = await adapter.requestDevice();

  // 4) Configure the canvas context
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'premultiplied' });

  // 5) One-off clear to color
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      loadOp: 'clear',
      clearValue: { r: 0.3921, g: 0.5843, b: 0.9294, a: 1 },
      storeOp: 'store'
    }]
  });
  pass.end();

  device.queue.submit([encoder.finish()]);
}

function showError(msg) {
  // Display on screen and console for quick debugging
  const pre = document.createElement('pre');
  pre.textContent = msg;
  pre.style.position = 'fixed';
  pre.style.top = '8px';
  pre.style.left = '8px';
  pre.style.padding = '8px';
  pre.style.background = 'rgba(0,0,0,0.75)';
  pre.style.color = '#fff';
  pre.style.zIndex = 9999;
  pre.style.whiteSpace = 'pre-wrap';
  document.body.appendChild(pre);
  console.error(msg);
}
