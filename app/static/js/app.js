// Function navigasi onepage application
function navigate(targetId) {
  const sections = document.querySelectorAll(".content-section");
  sections.forEach(section => {
    if (section.id === targetId) {
      section.classList.remove("hidden");
    } else {
      section.classList.add("hidden");
    }
  });

  if (targetId === 'preprocess') {
    loadOriginalSample(); 
  }
}

//Function Handle file upload
async function handleFileUpload() {
  const fileInput = document.getElementById('datasetFile');
  const uploadStatus = document.getElementById('uploadStatus');

  if (!fileInput || !uploadStatus) {
    console.error('Elemen input atau status tidak ditemukan');
    return;
  }

  const file = fileInput.files[0];

  if (!file) {
    uploadStatus.classList.remove('hidden', 'text-green-600', 'text-red-600');
    uploadStatus.classList.add('text-red-600');
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

    uploadStatus.classList.remove('hidden', 'text-red-600', 'text-green-600');

    if (response.ok) {
      uploadStatus.classList.add('text-green-600');
      uploadStatus.innerText = `✅ ${result.message || 'File berhasil diupload!'}`;
    } else {
      uploadStatus.classList.add('text-red-600');
      uploadStatus.innerText = `❌ ${result.error || 'Upload gagal.'}`;
    }
  } catch (error) {
    uploadStatus.classList.remove('hidden', 'text-green-600');
    uploadStatus.classList.add('text-red-600');
    uploadStatus.innerText = '❌ Terjadi kesalahan saat mengunggah file.';
    console.error(error);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const uploadBtn = document.getElementById('uploadBtn');
  const showBtn = document.getElementById('showBtn');
  const tableContainer = document.getElementById('tableContainer');

  if (!uploadBtn || !showBtn || !tableContainer) {
    console.error('Satu atau lebih elemen (#uploadBtn, #showBtn, #tableContainer) tidak ditemukan');
    return;
  }

  uploadBtn.addEventListener('click', handleFileUpload);

  // Show dataset button handling
  showBtn.addEventListener('click', async () => {
    tableContainer.innerHTML = '⏳ Memuat data...';

    try {
      const response = await fetch('/show_dataset');
      const result = await response.json();

      if (response.ok) {
        const { data, shape, columns } = result;

        if (!data || !columns || !shape) {
          tableContainer.innerText = '⚠️ Data tidak lengkap.';
          return;
        }

        let html = `<p class="mb-2">📄 Shape Data: ${shape.rows}, ${shape.columns}</p>`;
        html += '<table class="table-auto w-full border border-gray-300"><thead><tr>';

        columns.forEach(col => {
          html += `<th class="px-2 py-1 border bg-gray-100">${col}</th>`;
        });

        html += '</tr></thead><tbody>';

        data.forEach(row => {
          html += '<tr>';
          columns.forEach(col => {
            html += `<td class="px-2 py-1 border">${row[col]}</td>`;
          });
          html += '</tr>';
        });

        html += '</tbody></table>';
        tableContainer.innerHTML = html;
      } else {
        tableContainer.innerText = `❌ ${result.error || 'Gagal menampilkan data'}`;
      }
    } catch (error) {
      console.error(error);
      tableContainer.innerText = '❌ Terjadi kesalahan saat mengambil data.';
    }
  });
});

//Frontend Handling preprocess
// Function build table
function buildTable(data) {
  if (!data.length) return '';

  let html = '<table class="table-auto w-full border border-gray-300m">';
  html += '<thead><tr>';
  html += '<th class="border px-2 py-1 bg-gray-100">No.</th>';  // Tambahkan kolom No.
  html += '<th class="border px-2 py-1 bg-gray-100">Tweet</th>';
  html += '</tr></thead><tbody>';

  data.forEach((row, index) => {
    html += '<tr>';
    html += `<td class="border px-2 py-1">${index + 1}</td>`;           // No. urut
    html += `<td class="border px-2 py-1">${row['Tweet']}</td>`;       // Isi kolom Tweet
    html += '</tr>';
  });

  html += '</tbody></table>';
  return html;
}

async function loadOriginalSample() {
  const target = document.getElementById('originalData');
  target.innerHTML = '⏳ Memuat data asli…';

  try {
    const res = await fetch('/show_dataset');
    const json = await res.json();
    if (res.ok) {
      const { data, shape } = json;
      target.innerHTML = `<p>📄 Jumlah Baris: ${shape.rows}</p>` + buildTable(data);
    } else {
      target.innerHTML = `❌ ${json.error || 'Gagal memuat data'}`;
    }
  } catch (err) {
    console.error(err);
    target.innerHTML = '❌ Terjadi kesalahan saat memuat data.';
  }
}

document.getElementById('preprocessBtn').addEventListener('click', async () => {
  const preprocessedDataDiv = document.getElementById('preprocessedData');

  preprocessedDataDiv.innerHTML = '⏳ Memuat data hasil preprocessing...';

  try {
    const response = await fetch('/preprocess');
    const result = await response.json();

    if (response.ok) {
      const { processed, shape_processed } = result;
      preprocessedDataDiv.innerHTML = `<p>📄 Jumlah Baris: ${shape_processed[0]}</p>` + buildTable(processed);
    } else {
      preprocessedDataDiv.innerText = `❌ ${result.error || 'Gagal memuat data'}`;
    }
  } catch (error) {
    console.error(error);
    preprocessedDataDiv.innerText = '❌ Terjadi kesalahan saat memuat data.';
  }
});

//Frontend handling tokenization
document.getElementById('startTokenizeProgress').addEventListener('click', () => {
  const eventSource = new EventSource('/start-tokenization');
  const progressBar = document.getElementById('progressBar');
  const progressText = document.getElementById('progressPercent');
  const progressStatus = document.getElementById('progressStatus');
  const container = document.getElementById('progressContainer');

  container.classList.remove('hidden');
  progressBar.style.width = '0%';
  progressBar.className = 'h-4 bg-blue-600 rounded-full transition-all duration-300 ease-out';
  progressText.textContent = '0%';
  progressStatus.textContent = 'Memulai tokenisasi...';

  eventSource.onmessage = function(event) {
    const data = event.data.trim();

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
      eventSource.close();
    }
  };

  eventSource.onerror = () => {
    progressStatus.textContent = 'Terjadi kesalahan saat tokenisasi.';
    progressBar.classList.replace('bg-blue-600', 'bg-red-500');
    eventSource.close();
  };
});


document.getElementById('showTokenSample').addEventListener('click', async () => {
  const output     = document.getElementById('tokenResult');
  const preText    = document.getElementById('preTokenText');
  const tokenTitle = document.getElementById('tokenTitle');

  // state awal
  output.innerHTML      = 'Memuat...';
  preText.textContent   = 'Loading...';
  tokenTitle.textContent = 'Hasil Tokenisasi:';

  // helper jika gagal
  const handleError = (msg) => {
    output.innerHTML    = `<p class="text-red-500">${msg}</p>`;
    preText.textContent = '-';
    tokenTitle.textContent = 'Hasil Tokenisasi:';
  };

  try {
    const res  = await fetch('/tokenization-sample');
    const data = await res.json();

    if (data.error) {
      handleError(`Error: ${data.error}`);
      return;
    }

    const { text, tokens, token_ids, embedding_shape, embedding_preview } = data;

    preText.textContent   = text;
    tokenTitle.textContent = `Hasil Tokenisasi (embedding shape ${embedding_shape.join(', ')}):`;

    let tableHTML = `
      <table class="table-auto w-full border border-gray-300">
        <thead>
          <tr class="bg-gray-200">
            <th class="border px-2 py-1">#</th>
            <th class="border px-2 py-1">Token</th>
            <th class="border px-2 py-1">Token ID</th>
            <th class="border px-2 py-1">Embedding (Preview)</th>
          </tr>
        </thead>
        <tbody>
    `;

    tokens.forEach((tok, i) => {
      tableHTML += `
        <tr>
          <td class="border px-2 py-1 text-center">${i + 1}</td>
          <td class="border px-2 py-1">${tok}</td>
          <td class="border px-2 py-1 text-center">${token_ids[i]}</td>
          <td class="border px-2 py-1 text-xs">${embedding_preview[i]?.map(n => n.toFixed(4)).join(', ') || '-'}</td>
        </tr>`;
    });

    tableHTML += '</tbody></table>';
    output.innerHTML = tableHTML;

  } catch (err) {
    console.error(err);
    handleError('Gagal memuat hasil tokenisasi.');
  }
});


//Tuning
document.getElementById("btnTune").addEventListener("click", () => {
    const eventSource = new EventSource("/start-tuning");

    const logElement = document.getElementById("tuningLog");
    const tableBody = document.querySelector("#tune table tbody");
    const spinner = document.getElementById("tuningSpinner");

    logElement.innerText = "";
    tableBody.innerHTML = "";

    // Tampilkan spinner saat tuning mulai
    spinner.style.display = "inline-block";

    eventSource.onmessage = function(event) {
        const line = event.data;

        logElement.innerText += line + "\n";
        logElement.scrollTop = logElement.scrollHeight;

        if (line.includes("DONE")) {
            eventSource.close();
            // Delay untuk memberi waktu tulis file
            setTimeout(() => {
                fetch('/get-tuning-result')
                    .then(response => response.json())
                    .then(data => {
                        tableBody.innerHTML = "";
                        data.forEach(row => {
                            const tr = document.createElement("tr");
                            tr.innerHTML = `
                              <td class="px-4 py-2 text-center align-middle">${row.iteration}</td>
                              <td class="px-4 py-2 text-center align-middle">${row.params.epochs}</td>
                              <td class="px-4 py-2 text-center align-middle">${row.params.units}</td>
                              <td class="px-4 py-2 text-center align-middle">${row.params.learning_rate}</td>
                              <td class="px-4 py-2 text-center align-middle">${row.params.batch_size}</td>
                              <td class="px-4 py-2 text-center align-middle">${row.val_acc.toFixed(4)}</td>
                            `;
                            tableBody.appendChild(tr);
                        });
                        fetch('/get-best-params')
                            .then(res => res.json())
                            .then(data => {
                              if (data && Object.keys(data).length > 0) {
                                document.getElementById('epochCell').textContent = data.epochs;
                                document.getElementById('unitsCell').textContent = data.units;
                                document.getElementById('lrCell').textContent = data.learning_rate;
                                document.getElementById('bsCell').textContent = data.batch_size;
                              }
                            });
                    })
                    .catch(err => {
                        logElement.innerText += "\n❌ Gagal memuat hasil tuning: " + err.message;
                    })
                    .finally(() => {
                        // Sembunyikan spinner setelah selesai
                        spinner.style.display = "none";
                    });
            }, 1000);
        }
    };

    eventSource.onerror = function() {
        logElement.innerText += "\n❌ Terjadi kesalahan saat streaming data.";
        eventSource.close();
        spinner.style.display = "none";
    };
});

document.addEventListener('DOMContentLoaded', () => {
      const trainBtn = document.getElementById('trainBtn');
      const logArea = document.getElementById('logArea');
      const spinner = document.getElementById('spinner');

      // Load best parameters
      fetch('/get-best-params')
        .then(res => res.json())
        .then(data => {
          if (data && Object.keys(data).length > 0) {
            document.getElementById('epochCell').textContent = data.epochs;
            document.getElementById('unitsCell').textContent = data.units;
            document.getElementById('lrCell').textContent = data.learning_rate;
            document.getElementById('bsCell').textContent = data.batch_size;
          }
        });

      // Handle training button
      trainBtn.addEventListener('click', () => {
        logArea.textContent = "Memulai training...\n";
        spinner.classList.remove('hidden');

        const eventSource = new EventSource('/train-model');
        eventSource.onmessage = function (e) {
          logArea.textContent += e.data + "\n";
          logArea.scrollTop = logArea.scrollHeight;
          if (e.data.includes("Training selesai")) {
            spinner.classList.add('hidden');
            eventSource.close();
          }
        };
      });
    });

//Front end handling testing
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
        if (r.error) {
          resultsContainer.innerHTML = `<p class="text-red-600">${r.error}</p>`;
          return;
        }

        const div = document.createElement('div');
        div.classList.add('px-4', 'py-2', 'rounded', 'shadow', 'mb-2', 'transition', 'duration-300');

        if (r.prediction === 1) {
          div.classList.add('bg-red-100', 'text-red-800', 'font-semibold');
        } else {
          div.classList.add('bg-green-100', 'text-green-800');
        }

        div.innerHTML = `
          <p><strong>${r.label}</strong></p>
          <p>Probabilitas: <code>${r.probability?.toFixed(6) ?? 'N/A'}</code></p>
          <p>Prediksi Biner: <strong>${r.prediction}</strong></p>
        `;

        resultsContainer.appendChild(div);
      });
    } catch (error) {
      resultsContainer.innerHTML = `<p class="text-red-600">Model belum ada, silakan latih model terlebih dahulu.</p>
      <br>
      <p class="text-red-600">Error: ${error.message}</p>`;
    }
  });
});

//frontend handling evaluation
document.getElementById("runEvaluation").addEventListener("click", async () => {
  const button = document.getElementById("runEvaluation");
  const div = document.querySelector("#evaluate .mt-6");
  div.innerHTML = "";

  let statusText = document.createElement("span");
  statusText.id = "eval-status";
  statusText.className = "ml-2 text-600";
  statusText.innerText = "Sedang diproses...";
  button.parentNode.insertBefore(statusText, button.nextSibling);

  try {
    const response = await fetch("/evaluate-model");
    const data = await response.json();

    if (!response.ok || data.status !== "success") {
      const msg = data.message || "Terjadi kesalahan saat evaluasi.";
      div.innerHTML = `<p class="text-red-600">${msg.includes("Model") ? "Model belum ada, silakan melatih model agar dapat dievaluasi." : msg}</p>`;
      statusText.remove();
      return;
    }

    // Tambahkan judul Confusion Matrix
    const cmTitle = document.createElement("h3");
    cmTitle.innerText = "Confusion Matrix";
    cmTitle.className = "text-lg font-semibold text-center mt-4 mb-2";
    div.appendChild(cmTitle);

    // Tambahkan gambar Confusion Matrix
    const img = document.createElement("img");
    img.src = data.confusion_matrix;
    img.alt = "Confusion Matrix";
    img.className = "mx-auto mt-2 border rounded w-full max-w-4xl";
    div.appendChild(img);

    // Ambil mean accuracy & loss
    const summaryResp = await fetch("/evaluation/classification_report.json");
    const summaryData = await summaryResp.json();
    const averageEntry = summaryData.find(item => item.label === "average");

    const meanAccuracy = averageEntry?.mean_accuracy ?? "N/A";
    const meanLoss = averageEntry?.mean_loss ?? "N/A";

    // Tambahkan judul Classification Report
    const reportTitle = document.createElement("h3");
    reportTitle.innerText = "Classification Report";
    reportTitle.className = "text-lg font-semibold mt-6";
    div.appendChild(reportTitle);

    // Tambahkan teks accuracy & loss
    const metricP = document.createElement("p");
    metricP.className = "text-sm text-gray-800 mt-2 mb-1";
    metricP.innerHTML = `Test Accuracy: <strong>${meanAccuracy}</strong> &nbsp; | &nbsp; Test Loss: <strong>${meanLoss}</strong>`;
    div.appendChild(metricP);

    // Ambil classification report
    const reportResp = await fetch(data.report);
    const reportData = await reportResp.json();

    // Bangun tabel classification report
    let tableHtml = `
      <table class="table-auto mt-4 w-full border border-collapse border-gray-300">
        <thead>
          <tr class="bg-gray-200">
            <th class="border px-2 py-1">Label</th>
            <th class="border px-2 py-1">Accuracy</th>
            <th class="border px-2 py-1">Recall</th>
            <th class="border px-2 py-1">Precision</th>
          </tr>
        </thead>
        <tbody>
          ${reportData.map(row => `
            <tr>
              <td class="border px-2 py-1">${row.label}</td>
              <td class="border px-2 py-1">${parseFloat(row.accuracy).toFixed(4)}</td>
              <td class="border px-2 py-1">${parseFloat(row.recall).toFixed(4)}</td>
              <td class="border px-2 py-1">${parseFloat(row.precision).toFixed(4)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;

    div.innerHTML += tableHtml;
    
    // Ambil Euclidean data
    const euclideanResp = await fetch("/get-euclidean");
    const euclideanData = await euclideanResp.json();

    // Tambahkan judul
    const euclideanTitle = document.createElement("h3");
    euclideanTitle.innerText = "Euclidean Distance Table";
    euclideanTitle.className = "text-lg font-semibold mt-6";
    div.appendChild(euclideanTitle);

    // Hitung rata-rata
    const meanDistance = euclideanData.reduce((sum, row) => sum + parseFloat(row.Euclidean_Distance), 0) / euclideanData.length;

    const avgParagraph = document.createElement("p");
    avgParagraph.className = "text-sm text-gray-800 mt-1";
    avgParagraph.innerHTML = `Rata-rata Euclidean Distance: <strong>${meanDistance.toFixed(4)}</strong>`;
    div.appendChild(avgParagraph);

    // Buat tabel scrollable
    const tableWrapper = document.createElement("div");
    tableWrapper.className = "overflow-x-auto max-h-96 overflow-y-scroll border mt-2 rounded";

    let euclideanTable = `
      <table class="table-auto w-full border border-collapse border-gray-300">
        <thead class="bg-gray-200">
          <tr>
            <th class="border px-2 py-1">No.</th>
            <th class="border px-2 py-1">Tweet</th>
            <th class="border px-2 py-1">Euclidean Distance</th>
          </tr>
        </thead>
        <tbody>
          ${euclideanData.map((row, index) => `
            <tr>
              <td class="border px-2 py-1 text-center">${index + 1}</td>
              <td class="border px-2 py-1">${row.Tweet}</td>
              <td class="border px-2 py-1 text-center">${parseFloat(row.Euclidean_Distance).toFixed(4)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
    tableWrapper.innerHTML = euclideanTable;
    div.appendChild(tableWrapper);

  } catch (err) {
    div.innerHTML = `<p class="text-red-600">Model belum ada, silakan melatih model agar dapat dievaluasi.</p>`;
    console.error(err);
  }

  const existingStatus = document.getElementById("eval-status");
  if (existingStatus) existingStatus.remove();
});






