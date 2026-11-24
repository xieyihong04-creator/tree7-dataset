/* tree7 static demo: ONNX Runtime Web (single-thread wasm) + CAM overlay. */

const MODEL_URL = "model.onnx";
const CONFIG_URL = "config.json";
const IMG_SIZE = 224;

const state = {
  config: null,
  session: null,
  bitmap: null,
  busy: false,
  ready: false,
};

const el = {
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("file-input"),
  preview: null,
  runButton: document.getElementById("run-button"),
  hint: document.getElementById("hint"),
  statusLine: document.getElementById("status-line"),
  progressWrap: document.getElementById("progress-wrap"),
  progressBar: document.getElementById("progress-bar"),
  progressText: document.getElementById("progress-text"),
  empty: document.getElementById("empty-state"),
  result: document.getElementById("result"),
  predZh: document.getElementById("pred-zh"),
  predLatin: document.getElementById("pred-latin"),
  predConfidence: document.getElementById("pred-confidence"),
  bars: document.getElementById("bars"),
  cropCanvas: document.getElementById("crop-canvas"),
  camCanvas: document.getElementById("cam-canvas"),
  note: document.getElementById("note"),
  error: document.getElementById("error-box"),
  sampleList: document.getElementById("sample-list"),
  classTable: document.getElementById("class-table"),
};

function setStatus(text) {
  el.statusLine.textContent = text;
}

function setProgress(ratio, label) {
  if (ratio == null) {
    el.progressBar.style.width = "35%";
    el.progressBar.style.opacity = "0.55";
    el.progressText.textContent = label || "";
    return;
  }
  const pct = Math.max(0, Math.min(100, Math.round(ratio * 100)));
  el.progressBar.style.width = pct + "%";
  el.progressBar.style.opacity = "1";
  el.progressText.textContent = pct + "%";
}

function hideProgress() {
  el.progressWrap.style.display = "none";
}

function showError(message) {
  el.error.hidden = false;
  el.error.textContent = message;
}

function clearError() {
  el.error.hidden = true;
  el.error.textContent = "";
}

function escapeText(value) {
  return String(value);
}

function renderClasses() {
  const { class_zh: zh, classes } = state.config;
  el.classTable.innerHTML = "";
  classes.forEach((name, index) => {
    const tr = document.createElement("tr");
    const idx = document.createElement("td");
    idx.textContent = index + 1;
    const cn = document.createElement("td");
    cn.textContent = zh[index];
    const latin = document.createElement("td");
    latin.textContent = name;
    tr.append(idx, cn, latin);
    el.classTable.appendChild(tr);
  });
}

function renderSamples() {
  el.sampleList.innerHTML = "";
  state.config.examples.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sample";
    button.title = `辨識 ${item.class_zh} 範例`;
    const img = document.createElement("img");
    img.src = item.file;
    img.alt = `${item.class_zh} ${item.class}`;
    img.loading = "lazy";
    const span = document.createElement("span");
    span.textContent = item.class_zh;
    button.append(img, span);
    button.addEventListener("click", async () => {
      if (state.busy) return;
      try {
        const response = await fetch(item.file);
        const blob = await response.blob();
        await useFile(blob, item.class_zh);
      } catch (error) {
        showError("範例載入失敗：" + error.message);
      }
    });
    el.sampleList.appendChild(button);
  });
}

async function loadModel() {
  setStatus("正在下載量化模型（約 11 MB）…");
  try {
    const response = await fetch(MODEL_URL);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const total = Number(response.headers.get("content-length")) || 0;
    let buffer;
    if (!response.body) {
      buffer = await response.arrayBuffer();
      setProgress(1);
    } else {
      const reader = response.body.getReader();
      const chunks = [];
      let received = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        if (total) {
          setProgress(received / total);
        } else {
          setProgress(null, (received / 1048576).toFixed(1) + " MB");
        }
      }
      const blob = new Blob(chunks, { type: "application/octet-stream" });
      buffer = await blob.arrayBuffer();
    }

    setStatus("正在初始化推論引擎…");
    setProgress(null, "初始化");
    ort.env.wasm.wasmPaths = new URL("vendor/ort/", document.baseURI).href;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.logLevel = "error";

    state.session = await ort.InferenceSession.create(buffer, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    state.ready = true;
    hideProgress();
    el.hint.textContent = "模型就緒";
    setStatus("模型已就緒。上傳照片或點選左側範例。");
    if (state.bitmap) el.runButton.disabled = false;
  } catch (error) {
    hideProgress();
    el.hint.textContent = "模型載入失敗";
    setStatus("模型載入失敗。");
    showError("無法載入模型：" + error.message + "。請重新整理頁面再試。");
    console.error(error);
  }
}

function drawPreview(source) {
  let preview = el.dropzone.querySelector(".preview");
  if (!preview) {
    preview = document.createElement("img");
    preview.className = "preview";
    preview.alt = "已選取的圖片";
    el.dropzone.prepend(preview);
  }
  if (preview.dataset.url) URL.revokeObjectURL(preview.dataset.url);
  const url = URL.createObjectURL(source);
  preview.src = url;
  preview.dataset.url = url;
  el.dropzone.classList.add("has-image");
}

function preprocess(bitmap) {
  const scale = 255 / Math.min(bitmap.width, bitmap.height);
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));

  const resized = document.createElement("canvas");
  resized.width = width;
  resized.height = height;
  const rctx = resized.getContext("2d", { willReadFrequently: false });
  rctx.imageSmoothingEnabled = true;
  rctx.imageSmoothingQuality = "high";
  rctx.drawImage(bitmap, 0, 0, width, height);

  const left = Math.floor((width - IMG_SIZE) / 2);
  const top = Math.floor((height - IMG_SIZE) / 2);

  const crop = el.cropCanvas;
  const cctx = crop.getContext("2d", { willReadFrequently: true });
  cctx.clearRect(0, 0, IMG_SIZE, IMG_SIZE);
  cctx.drawImage(resized, left, top, IMG_SIZE, IMG_SIZE, 0, 0, IMG_SIZE, IMG_SIZE);

  const pixels = cctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data;
  const tensor = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
  const mean = state.config.mean;
  const std = state.config.std;
  const plane = IMG_SIZE * IMG_SIZE;
  for (let i = 0; i < plane; i += 1) {
    const r = pixels[i * 4] / 255;
    const g = pixels[i * 4 + 1] / 255;
    const b = pixels[i * 4 + 2] / 255;
    tensor[i] = (r - mean[0]) / std[0];
    tensor[plane + i] = (g - mean[1]) / std[1];
    tensor[2 * plane + i] = (b - mean[2]) / std[2];
  }
  return { tensor, pixels };
}

function softmax(values) {
  let max = -Infinity;
  for (const value of values) max = Math.max(max, value);
  const exps = new Float64Array(values.length);
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    const value = Math.exp(values[i] - max);
    exps[i] = value;
    sum += value;
  }
  for (let i = 0; i < exps.length; i += 1) exps[i] /= sum;
  return exps;
}

function jet(value) {
  const clamp = (x) => Math.max(0, Math.min(1, x));
  return [
    clamp(1.5 - Math.abs(4 * value - 3)) * 255,
    clamp(1.5 - Math.abs(4 * value - 2)) * 255,
    clamp(1.5 - Math.abs(4 * value - 1)) * 255,
  ];
}

function buildCam(features, classIndex, basePixels) {
  const weights = state.config.fc_weight[classIndex];
  const cells = 49;
  const cam = new Float32Array(cells);
  for (let channel = 0; channel < weights.length; channel += 1) {
    const weight = weights[channel];
    if (!weight) continue;
    const offset = channel * cells;
    for (let cell = 0; cell < cells; cell += 1) {
      cam[cell] += weight * features[offset + cell];
    }
  }
  let max = 0;
  for (let cell = 0; cell < cells; cell += 1) {
    cam[cell] = Math.max(0, cam[cell]);
    max = Math.max(max, cam[cell]);
  }
  if (max > 0) {
    for (let cell = 0; cell < cells; cell += 1) cam[cell] /= max;
  }

  const small = document.createElement("canvas");
  small.width = 7;
  small.height = 7;
  const sctx = small.getContext("2d");
  const smallImage = sctx.createImageData(7, 7);
  for (let cell = 0; cell < cells; cell += 1) {
    const [r, g, b] = jet(cam[cell]);
    smallImage.data[cell * 4] = r;
    smallImage.data[cell * 4 + 1] = g;
    smallImage.data[cell * 4 + 2] = b;
    smallImage.data[cell * 4 + 3] = 255;
  }
  sctx.putImageData(smallImage, 0, 0);

  const heat = document.createElement("canvas");
  heat.width = IMG_SIZE;
  heat.height = IMG_SIZE;
  const hctx = heat.getContext("2d");
  hctx.imageSmoothingEnabled = true;
  hctx.imageSmoothingQuality = "high";
  hctx.drawImage(small, 0, 0, IMG_SIZE, IMG_SIZE);
  const heatPixels = hctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data;

  const out = el.camCanvas.getContext("2d").createImageData(IMG_SIZE, IMG_SIZE);
  for (let i = 0; i < basePixels.length; i += 4) {
    out.data[i] = 0.58 * basePixels[i] + 0.42 * heatPixels[i];
    out.data[i + 1] = 0.58 * basePixels[i + 1] + 0.42 * heatPixels[i + 1];
    out.data[i + 2] = 0.58 * basePixels[i + 2] + 0.42 * heatPixels[i + 2];
    out.data[i + 3] = 255;
  }
  el.camCanvas.getContext("2d").putImageData(out, 0, 0);
}

function renderPrediction(probs, classIndex) {
  const { class_zh: zh, classes } = state.config;
  el.predZh.textContent = zh[classIndex];
  el.predLatin.textContent = classes[classIndex];
  el.predConfidence.textContent = (probs[classIndex] * 100).toFixed(1) + "%";

  const order = Array.from(probs.keys())
    .sort((a, b) => probs[b] - probs[a])
    .slice(0, 3);

  el.bars.innerHTML = "";
  order.forEach((index, rank) => {
    const row = document.createElement("div");
    row.className = "bar-row" + (rank === 0 ? " top" : "");
    const name = document.createElement("div");
    name.className = "bar-name";
    name.textContent = zh[index];
    const small = document.createElement("small");
    small.textContent = classes[index];
    name.appendChild(small);
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    track.appendChild(fill);
    const value = document.createElement("div");
    value.className = "bar-value";
    value.textContent = (probs[index] * 100).toFixed(1) + "%";
    row.append(name, track, value);
    el.bars.appendChild(row);
    requestAnimationFrame(() => {
      fill.style.width = (probs[index] * 100).toFixed(2) + "%";
    });
  });

  const top = probs[classIndex];
  if (top < 0.5) {
    el.note.className = "note warn";
    el.note.textContent =
      "信心偏低：影像可能不是這 7 種樹，或葉片與樹冠特徵不足。建議換一張主體更清楚的照片。";
  } else {
    el.note.className = "note";
    el.note.textContent =
      "熱區偏紅代表模型判讀時最依賴的位置。若熱區落在背景或周圍植被，表示模型可能受到場景線索影響。";
  }
}

async function run() {
  if (!state.bitmap || state.busy) return;
  if (!state.ready) {
    el.hint.textContent = "模型還沒載入完成";
    return;
  }
  state.busy = true;
  el.runButton.disabled = true;
  el.hint.textContent = "推論中…";
  clearError();
  await new Promise((resolve) => setTimeout(resolve, 20));

  try {
    const prepared = preprocess(state.bitmap);
    const inputName = state.session.inputNames[0];
    const feeds = {
      [inputName]: new ort.Tensor("float32", prepared.tensor, [1, 3, IMG_SIZE, IMG_SIZE]),
    };
    const started = performance.now();
    const output = await state.session.run(feeds);
    const elapsed = Math.round(performance.now() - started);

    const logitsTensor = output.logits || output[state.session.outputNames[0]];
    const featureTensor = output.features || output[state.session.outputNames[1]];
    const probs = softmax(logitsTensor.data);
    let best = 0;
    for (let i = 1; i < probs.length; i += 1) {
      if (probs[i] > probs[best]) best = i;
    }

    renderPrediction(probs, best);
    buildCam(featureTensor.data, best, prepared.pixels);

    el.empty.hidden = true;
    el.result.hidden = false;
    setStatus(`完成 · ${elapsed} ms · 全部在瀏覽器內執行`);
    el.hint.textContent = "模型就緒";
  } catch (error) {
    showError("推論失敗：" + error.message);
    setStatus("推論失敗。");
    console.error(error);
  } finally {
    state.busy = false;
    el.runButton.disabled = !state.bitmap;
  }
}

async function useFile(file, label) {
  if (!file || !file.type || !file.type.startsWith("image/")) {
    showError("請選擇圖片檔案（JPG、PNG、WebP 等）。");
    return;
  }
  clearError();
  try {
    const bitmap = await createImageBitmap(file);
    state.bitmap = bitmap;
    drawPreview(file);
    el.empty.hidden = false;
    el.result.hidden = true;
    el.runButton.disabled = !state.ready;
    setStatus(label ? `已載入範例：${label}` : "圖片已載入，開始辨識中…");
    await run();
  } catch (error) {
    showError("無法讀取圖片：" + error.message);
  }
}

function setupEvents() {
  el.fileInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (file) useFile(file);
    event.target.value = "";
  });

  el.runButton.addEventListener("click", () => run());

  ["dragenter", "dragover"].forEach((name) => {
    el.dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      el.dropzone.classList.add("dragging");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    el.dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      el.dropzone.classList.remove("dragging");
    });
  });
  el.dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) useFile(file);
  });

  window.addEventListener("paste", (event) => {
    const items = event.clipboardData && event.clipboardData.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) {
          useFile(file);
          break;
        }
      }
    }
  });
}

async function init() {
  setupEvents();
  try {
    const response = await fetch(CONFIG_URL);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.config = await response.json();
    renderClasses();
    renderSamples();
  } catch (error) {
    setStatus("設定檔載入失敗。");
    showError("無法載入 config.json：" + error.message);
    return;
  }
  await loadModel();
}

init();
