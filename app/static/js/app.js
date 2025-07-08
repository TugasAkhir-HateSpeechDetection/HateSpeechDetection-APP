// Navigasi antar halaman di aplikasi one-page
function navigate(targetId) {
  document.querySelectorAll(".content-section").forEach(section => {
    section.classList.toggle("hidden", section.id !== targetId);
  });
  if (targetId === 'preprocess') loadOriginalSample();
}

// Proses upload file dataset .csv ke server
async function handleFileUpload() {
  const fileInput = document.getElementById('datasetFile');
  const uploadStatus = document.getElementById('uploadStatus');
  const file = fileInput?.files[0];

  if (!file) {
    uploadStatus.className = 'mt-4 text-red-600 font-medium';
    uploadStatus.innerText = '❌ Silakan pilih file terlebih dahulu.';
    return;
  }

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('/upload_dataset', {
      method: 'POST',
      body: formData,
    });

    const result = await response.json();
    uploadStatus.className = response.ok ? 'mt-4 text-green-600 font-medium' : 'mt-4 text-red-600 font-medium';
    uploadStatus.innerText = response.ok
      ? `✅ ${result.message || 'File berhasil diunggah.'}`
      : `❌ ${result.error || 'Upload gagal.'}`;
  } catch (error) {
    console.error(error);
    uploadStatus.className = 'mt-4 text-red-600 font-medium';
    uploadStatus.innerText = '❌ Terjadi kesalahan saat mengunggah file.';
  }
}

// Build HTML table dari data tweet
function buildTable(data) {
  if (!data.length) return '';
  let html = '<table class="table-auto w-full border border-gray-300"><thead><tr>';
  html += '<th class="border px-2 py-1 bg-gray-100">No.</th>';
  html += '<th class="border px-2 py-1 bg-gray-100">Tweet</th>';
  html += '</tr></thead><tbody>';
  data.forEach((row, i) => {
    html += `<tr><td class="border px-2 py-1">${i + 1}</td><td class="border px-2 py-1">${row.Tweet}</td></tr>`;
  });
  html += '</tbody></table>';
  return html;
}

// Tampilkan data asli sebelum preprocessing
async function loadOriginalSample() {
  const target = document.getElementById('originalData');
  target.innerHTML = '⏳ Memuat data asli…';
  try {
    const res = await fetch('/show_dataset');
    const json = await res.json();
    target.innerHTML = res.ok ? `<p>📄 Jumlah Baris: ${json.shape.rows}</p>` + buildTable(json.data)
                              : `❌ ${json.error || 'Gagal memuat data'}`;
  } catch (err) {
    console.error(err);
    target.innerHTML = '❌ Terjadi kesalahan saat memuat data.';
  }
}

// Proses event setelah halaman dimuat
document.addEventListener('DOMContentLoaded', () => {
  // Upload dataset
  document.getElementById('uploadBtn')?.addEventListener('click', handleFileUpload);

  // Tampilkan isi dataset
  document.getElementById('showBtn')?.addEventListener('click', async () => {
    const container = document.getElementById('tableContainer');
    container.innerHTML = '⏳ Memuat data...';
    try {
      const res = await fetch('/show_dataset');
      const result = await res.json();
      if (res.ok) {
        const { data, shape, columns } = result;
        let html = `<p class="mb-2">📄 Shape Data: ${shape.rows}, ${shape.columns}</p>`;
        html += '<table class="table-auto w-full border border-gray-300"><thead><tr>';
        columns.forEach(col => html += `<th class="px-2 py-1 border bg-gray-100">${col}</th>`);
        html += '</tr></thead><tbody>';
        data.forEach(row => {
          html += '<tr>' + columns.map(col => `<td class="px-2 py-1 border">${row[col]}</td>`).join('') + '</tr>';
        });
        html += '</tbody></table>';
        container.innerHTML = html;
      } else {
        container.innerText = `❌ ${result.error || 'Gagal menampilkan data'}`;
      }
    } catch (err) {
      container.innerText = '❌ Terjadi kesalahan saat mengambil data.';
    }
  });

  // Preprocessing
  document.getElementById('preprocessBtn')?.addEventListener('click', async () => {
    const target = document.getElementById('preprocessedData');
    target.innerHTML = '⏳ Memuat data hasil preprocessing...';
    try {
      const res = await fetch('/preprocess');
      const result = await res.json();
      target.innerHTML = res.ok
        ? `<p>📄 Jumlah Baris: ${result.shape_processed[0]}</p>` + buildTable(result.processed)
        : `❌ ${result.error || 'Gagal memuat data'}`;
    } catch (err) {
      console.error(err);
      target.innerHTML = '❌ Terjadi kesalahan saat memuat data.';
    }
  });

  // Tokenisasi dengan progres bar (EventSource)
  document.getElementById('startTokenizeProgress')?.addEventListener('click', () => {
    const es = new EventSource('/start-tokenization');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressPercent');
    const progressStatus = document.getElementById('progressStatus');
    const container = document.getElementById('progressContainer');
    const barTitle = document.getElementById('barTitle');

    container.classList.remove('hidden');
    progressBar.style.width = '0%';
    progressBar.className = 'h-4 bg-blue-600 rounded-full transition-all duration-300 ease-out';
    progressText.textContent = '0%';
    progressStatus.textContent = 'Memulai tokenisasi...';

    es.onmessage = (e) => {
      const data = e.data.trim();
      if (data === 'ALREADY_EXISTS') {
        progressBar.style.width = '100%';
        progressBar.classList.replace('bg-blue-600', 'bg-green-500');
        progressText.textContent = '100%';
        progressStatus.textContent = 'Embedding sudah ada. Tidak perlu tokenisasi.';
        barTitle.classList.replace('text-blue-700', 'text-green-600');
        progressText.classList.replace('text-blue-700', 'text-green-600');
        es.close();
        return;
      }

      const match = data.match(/Tokenizing:\s*(\d+)%/);
      
      if (match) {
        const percent = parseInt(match[1]);
        progressBar.style.width = percent + '%';
        progressText.textContent = percent + '%';
        progressStatus.textContent = 'Memproses tweet...';
      }

      if (data.includes('DONE')) {
        progressBar.style.width = '100%';
        progressBar.classList.replace('bg-blue-600', 'bg-green-500');
        progressText.textContent = '100%';
        progressStatus.textContent = 'Tokenisasi selesai!';
        barTitle.classList.replace('text-blue-700', 'text-green-600');
        progressText.classList.replace('text-blue-700', 'text-green-600');
        es.close();
      }
    };

    es.onerror = () => {
      progressStatus.textContent = '❌ Terjadi kesalahan saat tokenisasi.';
      progressBar.classList.replace('bg-blue-600', 'bg-red-500');
      es.close();
    };
  });

  // Tampilkan sampel hasil tokenisasi
  document.getElementById('showTokenSample')?.addEventListener('click', async () => {
    const output = document.getElementById('tokenResult');
    const preText = document.getElementById('preTokenText');
    const tokenTitle = document.getElementById('tokenTitle');

    output.innerHTML = 'Memuat...';
    preText.textContent = 'Loading...';
    tokenTitle.textContent = 'Hasil Tokenisasi:';

    try {
      const res = await fetch('/tokenization-sample');
      const data = await res.json();

      if (data.error) {
        output.innerHTML = `<p class="text-red-500">${data.error}</p>`;
        preText.textContent = '-';
        return;
      }

      preText.textContent = data.text;
      tokenTitle.textContent = `Hasil Tokenisasi (embedding shape ${data.embedding_shape.join(', ')}):`;

      let tableHTML = `
        <table class="table-auto w-full border border-gray-300">
          <thead class="bg-gray-200">
            <tr>
              <th class="border px-2 py-1">#</th>
              <th class="border px-2 py-1">Token</th>
              <th class="border px-2 py-1">Token ID</th>
              <th class="border px-2 py-1">Embedding (Preview)</th>
            </tr>
          </thead>
          <tbody>
      `;

      data.tokens.forEach((tok, i) => {
        tableHTML += `
          <tr>
            <td class="border px-2 py-1 text-center">${i + 1}</td>
            <td class="border px-2 py-1">${tok}</td>
            <td class="border px-2 py-1 text-center">${data.token_ids[i]}</td>
            <td class="border px-2 py-1 text-xs">${data.embedding_preview[i]?.map(n => n.toFixed(4)).join(', ') || '-'}</td>
          </tr>
        `;
      });

      tableHTML += '</tbody></table>';
      output.innerHTML = tableHTML;

    } catch (err) {
      console.error(err);
      output.innerHTML = '<p class="text-red-500">Gagal memuat hasil tokenisasi.</p>';
    }
  });
});

// Event untuk tuning hyperparameter
document.getElementById("btnTune").addEventListener("click", () => {
  const eventSource = new EventSource("/start-tuning");
  const logElement = document.getElementById("tuningLog");
  const tableBody = document.querySelector("#tune table tbody");
  const spinner = document.getElementById("tuningSpinner");

  logElement.innerText = "";
  tableBody.innerHTML = "";
  spinner.style.display = "inline-block";

  eventSource.onmessage = function (event) {
    const line = event.data;
    logElement.innerText += line + "\n";
    logElement.scrollTop = logElement.scrollHeight;

    if (line.includes("DONE")) {
      eventSource.close();

      setTimeout(() => {
        fetch('/get-tuning-result')
          .then(res => res.json())
          .then(data => {
            tableBody.innerHTML = "";
            data.slice(0, 5).forEach(row => {
              const tr = document.createElement("tr");
              tr.innerHTML = `
                <td class="px-4 py-2 text-center">${row.iteration}</td>
                <td class="px-4 py-2 text-center">${row.params.epochs}</td>
                <td class="px-4 py-2 text-center">${row.params.units}</td>
                <td class="px-4 py-2 text-center">${row.params.learning_rate}</td>
                <td class="px-4 py-2 text-center">${row.params.batch_size}</td>
                <td class="px-4 py-2 text-center">${row.val_acc.toFixed(4)}</td>`;
              tableBody.appendChild(tr);
            });

            return fetch('/get-best-params');
          })
          .then(res => res.json())
          .then(params => {
            if (params && Object.keys(params).length) {
              document.getElementById('epochCell').textContent = params.epochs;
              document.getElementById('unitsCell').textContent = params.units;
              document.getElementById('lrCell').textContent = params.learning_rate;
              document.getElementById('bsCell').textContent = params.batch_size;
            }
          })
          .finally(() => {
            spinner.style.display = "none";
          });
      }, 1000);
    }
  };

  eventSource.onerror = () => {
    logElement.innerText += "\n❌ Terjadi kesalahan saat streaming data.";
    eventSource.close();
    spinner.style.display = "none";
  };
});

// Proses pelatihan model
document.addEventListener('DOMContentLoaded', () => {
  const trainBtn = document.getElementById('trainBtn');
  const logArea = document.getElementById('logArea');
  const spinner = document.getElementById('spinner');
  const showPlotBtn = document.getElementById('showPlotBtn');
  const showPlotWrap = document.getElementById('showPlotWrapper');
  const plotContainer = document.getElementById('trainingPlotContainer');
  const plotImg = document.getElementById('trainingPlot');

  // Ambil parameter terbaik
  fetch('/get-best-params')
    .then(res => res.json())
    .then(p => {
      if (p && Object.keys(p).length) {
        epochCell.textContent = p.epochs;
        unitsCell.textContent = p.units;
        lrCell.textContent = p.learning_rate;
        bsCell.textContent = p.batch_size;
      }
    });

  trainBtn.addEventListener('click', () => {
    logArea.textContent = 'Memulai training...\n';
    spinner.classList.remove('hidden');
    showPlotWrap.classList.add('hidden');
    plotContainer.classList.add('hidden');

    const es = new EventSource('/train-model');
    es.onmessage = (e) => {
      logArea.textContent += e.data + '\n';
      logArea.scrollTop = logArea.scrollHeight;

      if (e.data.includes('Training selesai')) {
        spinner.classList.add('hidden');
        showPlotWrap.classList.remove('hidden');
        es.close();
      }
    };
  });

  showPlotBtn.addEventListener('click', () => {
    const url = '/evaluation/training_plot.png?' + Date.now();
    const tmp = new Image();
    tmp.onload = () => {
      plotImg.src = url;
      plotContainer.classList.remove('hidden');
    };
    tmp.onerror = () =>
      alert('Gagal memuat training_plot.png – pastikan training selesai tanpa error.');
    tmp.src = url;
  });
});

// Proses prediksi teks tweet atau testing
document.addEventListener('DOMContentLoaded', () => {
  const button = document.querySelector('#test button');
  const textarea = document.querySelector('#test textarea');
  const resultsContainer = document.querySelector('#prediction-results');

  button.addEventListener('click', async () => {
    const tweet = textarea.value.trim();
    resultsContainer.innerHTML = '';

    if (!tweet) {
      resultsContainer.innerHTML = '<p class="text-red-600">Mohon masukkan tweet terlebih dahulu.</p>';
      return;
    }

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tweet })
      });

      const results = await response.json();

      if (!response.ok || results.error) {
        const msg = results.error || 'Terjadi kesalahan saat melakukan prediksi.';
        resultsContainer.innerHTML = `<p class="text-red-600">${msg.includes('Model') ? 'Model belum ada, silakan latih model terlebih dahulu.' : msg}</p>`;
        return;
      }

      results.forEach(r => {
        const div = document.createElement('div');
        div.className = `px-4 py-2 rounded shadow mb-2 ${r.prediction === 1 ? 'bg-red-100 text-red-800 font-semibold' : 'bg-green-100 text-green-800'}`;
        div.innerHTML = `
          <p><strong>${r.label}</strong></p>
          <p>Probabilitas: <code>${r.probability?.toFixed(6) ?? 'N/A'}</code></p>
          <p>Prediksi Biner: <strong>${r.prediction}</strong></p>`;
        resultsContainer.appendChild(div);
      });
    } catch (error) {
      resultsContainer.innerHTML = `<p class="text-red-600">Model belum ada, silakan latih model terlebih dahulu.</p><p>Error: ${error.message}</p>`;
    }
  });
});

// Evaluasi model (confusion matrix + classification report + hamming loss)
document.getElementById("runEvaluation").addEventListener("click", async () => {
  const button = document.getElementById("runEvaluation");
  const div = document.querySelector("#evaluate .mt-6");
  div.innerHTML = "";

  const statusText = document.createElement("span");
  statusText.className = "ml-2 text-gray-600";
  statusText.innerText = "Sedang diproses...";
  button.parentNode.insertBefore(statusText, button.nextSibling);

  try {
    const res = await fetch("/evaluate-model");
    const data = await res.json();
    if (!res.ok || data.status !== "success") {
      div.innerHTML = `<p class="text-red-600">${data.message || 'Terjadi kesalahan saat evaluasi.'}</p>`;
      statusText.remove();
      return;
    }

    // Confusion Matrix
    div.innerHTML += `
      <h3 class="text-lg font-semibold text-center mt-4 mb-2">Confusion Matrix</h3>
      <img src="${data.confusion_matrix}" alt="Confusion Matrix" class="mx-auto mt-2 border rounded w-full max-w-4xl">
    `;

    // Ambil mean accuracy & loss
    const summaryResp = await fetch("/evaluation/evaluation_score.json");
    const summaryData = await summaryResp.json();
    const evaluation = summaryData[0];

    const meanAccuracy = evaluation?.mean_accuracy ?? "N/A";
    const meanLoss = evaluation?.mean_loss ?? "N/A";

    // Classification Report
    div.innerHTML += `
      <h3 class="text-lg font-semibold mt-6">Classification Report</h3>
      <p class="text-sm text-gray-800 mt-2 mb-1">Test Accuracy: <strong>${(parseFloat(meanAccuracy) * 100).toFixed(2)}%</strong> &nbsp; | &nbsp; Test Loss: <strong>${meanLoss}</strong></p>
    `;

    const report = await (await fetch(data.report)).json();
    div.innerHTML += `
      <table class="table-auto mt-4 w-full border border-collapse border-gray-300">
        <thead class="bg-gray-200">
          <tr>
            <th class="border px-2 py-1">Label</th>
            <th class="border px-2 py-1">Accuracy</th>
            <th class="border px-2 py-1">Recall</th>
            <th class="border px-2 py-1">Precision</th>
          </tr>
        </thead>
        <tbody>
          ${report.map(row => `
            <tr>
              <td class="border px-2 py-1">${row.label}</td>
              <td class="border px-2 py-1">${parseFloat(row.accuracy).toFixed(4)}</td>
              <td class="border px-2 py-1">${parseFloat(row.recall).toFixed(4)}</td>
              <td class="border px-2 py-1">${parseFloat(row.precision).toFixed(4)}</td>
            </tr>`).join('')}
        </tbody>
      </table>
    `;

    // Hamming Loss Table
    const hammingData = await (await fetch("/get-hamming-loss")).json();
    const meanHamming = hammingData.reduce((sum, row) => sum + parseFloat(row.Hamming_Loss), 0) / hammingData.length;
    const similarityScore = 1 - meanHamming;
    div.innerHTML += `
      <h3 class="text-lg font-semibold mt-6">Hamming Loss Table</h3>
      <p class="text-sm text-gray-800 mt-2 mb-1">Similarity Score: <strong>${(similarityScore * 100).toFixed(2)}%</strong> | Mean Hamming Loss: <strong>${meanHamming.toFixed(4)}</strong></p>
      <div class="overflow-x-auto max-h-96 overflow-y-scroll border mt-2 rounded">
        <table class="table-auto w-full border border-collapse border-gray-300">
          <thead class="bg-gray-200">
            <tr>
              <th class="border px-2 py-1">No.</th>
              <th class="border px-2 py-1">Tweet</th>
              <th class="border px-2 py-1">Hamming Loss</th>
            </tr>
          </thead>
          <tbody>
            ${hammingData.map((row, i) => `
              <tr>
                <td class="border px-2 py-1 text-center">${i + 1}</td>
                <td class="border px-2 py-1">${row.Tweet}</td>
                <td class="border px-2 py-1 text-center">${parseFloat(row.Hamming_Loss).toFixed(4)}</td>
              </tr>`).join('')}
          </tbody>
        </table>
      </div>
    `;
  } catch (err) {
    console.error(err);
    div.innerHTML = `<p class="text-red-600">❌ Terjadi kesalahan saat evaluasi.</p>`;
  } finally {
    statusText.remove();
  }
});

