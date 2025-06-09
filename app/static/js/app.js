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

//Event listener
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

        let html = `<p class="mb-2">📄 Baris: ${shape.rows}, ${shape.columns}</p>`;
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
document.getElementById('preprocessBtn').addEventListener('click', async () => {
  const originalDataDiv = document.getElementById('originalData');
  const preprocessedDataDiv = document.getElementById('preprocessedData');

  originalDataDiv.innerHTML = '⏳ Memuat data asli...';
  preprocessedDataDiv.innerHTML = '⏳ Memuat data hasil preprocessing...';

  try {
    const response = await fetch('/preprocess');
    const result = await response.json();

    if (response.ok) {
      const { original, processed, shape_original, shape_processed } = result;

      const buildTable = (data) => {
        let html = '';
        if (data.length > 0) {
          html += '<table class="table-auto border border-gray-300 text-sm"><thead><tr>';
          Object.keys(data[0]).forEach(key => {
            html += `<th class="border px-2 py-1 bg-gray-100">${key}</th>`;
          });
          html += '</tr></thead><tbody>';
          data.forEach(row => {
            html += '<tr>';
            Object.values(row).forEach(val => {
              html += `<td class="border px-2 py-1">${val}</td>`;
            });
            html += '</tr>';
          });
          html += '</tbody></table>';
        }
        return html;
      };

      originalDataDiv.innerHTML = `<p>📄 Baris: ${shape_original[0]}</p>` + buildTable(original);
      preprocessedDataDiv.innerHTML = `<p>📄 Baris: ${shape_processed[0]}</p>` + buildTable(processed);
    } else {
      originalDataDiv.innerText = `❌ ${result.error || 'Gagal memuat data'}`;
      preprocessedDataDiv.innerText = '';
    }
  } catch (error) {
    console.error(error);
    originalDataDiv.innerText = '❌ Terjadi kesalahan saat memuat data.';
    preprocessedDataDiv.innerText = '';
  }
});

//Frontend handling tokenization
document.getElementById('tokenizeBtn').addEventListener('click', async () => {
  const output = document.getElementById('tokenResult');
  const preText = document.getElementById('preTokenText');

  output.innerHTML = '<p class="text-gray-500">Memproses...</p>';
  preText.textContent = 'Memuat...';

  try {
    const response = await fetch('/run_tokenization', { method: 'POST' });
    const data = await response.json();

    if (data.error) {
      output.innerHTML = `<p class="text-red-500">Error: ${data.error}</p>`;
      preText.textContent = '-';
      return;
    }

    const { text, tokens, token_ids, attention_mask, embedding_shape, embedding_preview } = data.token_data;
    preText.textContent = text;
    
    // Buat tabel
    let tableHTML = `
      <table class="table-auto border-collapse w-full text-sm">
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

    for (let i = 0; i < tokens.length; i++) {
      tableHTML += `
        <tr>
          <td class="border px-2 py-1 text-center">${i + 1}</td>
          <td class="border px-2 py-1">${tokens[i]}</td>
          <td class="border px-2 py-1 text-center">${token_ids[i]}</td>
          <td class="border px-2 py-1 text-xs">${embedding_preview[i] ? embedding_preview[i].map(n => n.toFixed(4)).join(', ') : '-'}</td>
        </tr>
      `;
    }

    tableHTML += `
        </tbody>
      </table>
      <p class="mt-4 text-gray-700"><strong>Embedding shape:</strong> (${embedding_shape.join(', ')})</p>
    `;

    output.innerHTML = tableHTML;

  } catch (err) {
    output.innerHTML = `<p class="text-red-500">Gagal memproses tokenisasi.</p>`;
    preText.textContent = '-';
    console.error(err);
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

    eventSource.onerror = function(event) {
    logElement.innerText += `\n Terjadi kesalahan saat streaming data.`;
    if (event && event.target && event.target.readyState === EventSource.CLOSED) {
        logElement.innerText += "\n Koneksi EventSource ditutup.";
    } else {
        logElement.innerText += `\n Detail Error: ${JSON.stringify(event)}`;
    }

    console.error("EventSource error:", event);
    eventSource.close();
    spinner.style.display = "none";
};
});

//Train
// document.addEventListener('DOMContentLoaded', () => {
//   const trainBtn = document.getElementById('trainBtn');
//   const logArea = document.getElementById('logArea');

//   // Load best parameters
//   fetch('/get-best-params')
//     .then(res => res.json())
//     .then(data => {
//       if (data && Object.keys(data).length > 0) {
//         document.getElementById('lrCell').textContent = data.learning_rate;
//         document.getElementById('bsCell').textContent = data.batch_size;
//         document.getElementById('epochCell').textContent = data.epochs;
//         document.getElementById('unitsCell').textContent = data.units;
//       }
//     });

//   // Training
//   trainBtn.addEventListener('click', () => {
//     logArea.textContent = "Memulai training...\n";
//     const eventSource = new EventSource('/train-model');
//     eventSource.onmessage = function (e) {
//       logArea.textContent += e.data + "\n";
//       logArea.scrollTop = logArea.scrollHeight;
//       if (e.data.includes("Training selesai")) {
//         eventSource.close();
//       }
//     };
//   });
// });

document.addEventListener('DOMContentLoaded', () => {
      const trainBtn = document.getElementById('trainBtn');
      const logArea = document.getElementById('logArea');
      const spinner = document.getElementById('spinner');

      // Load best parameters
      fetch('/get-best-params')
        .then(res => res.json())
        .then(data => {
          if (data && Object.keys(data).length > 0) {
            document.getElementById('lrCell').textContent = data.learning_rate;
            document.getElementById('bsCell').textContent = data.batch_size;
            document.getElementById('epochCell').textContent = data.epochs;
            document.getElementById('unitsCell').textContent = data.units;
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

//testing
document.addEventListener('DOMContentLoaded', () => {
  const button = document.querySelector('#test button');
  const textarea = document.querySelector('#test textarea');
  const resultsContainer = document.querySelector('#prediction-results');

  button.addEventListener('click', async () => {
    const tweet = textarea.value.trim();
    resultsContainer.innerHTML = ''; // Kosongkan hasil sebelumnya

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

      if (!response.ok) {
        throw new Error('Gagal mendapatkan response dari server');
      }

      const results = await response.json();
      if (results.error) {
        resultsContainer.innerHTML = `<p class="text-red-600">${results.error}</p>`;
        return;
      }

      results.forEach(r => {
        const div = document.createElement('div');
        div.classList.add('px-4', 'py-2', 'rounded', 'shadow', 'mb-2', 'transition', 'duration-300');

        // Warna latar dan teks
        if (r.prediction === 1) {
          div.classList.add('bg-red-100', 'text-red-800', 'font-semibold');
        } else if (r.probability >= 0.4) {
          div.classList.add('bg-yellow-100', 'text-yellow-800');
        } else {
          div.classList.add('bg-green-100', 'text-green-800');
        }


        div.innerHTML = `
          <p><strong>${r.label}</strong></p>
          <p>Probabilitas: <code>${r.probability.toFixed(6)}</code></p>
          <p>Prediksi Biner: <strong>${r.prediction}</strong></p>
        `;

        resultsContainer.appendChild(div);
      });

    } catch (error) {
      resultsContainer.innerHTML = `<p class="text-red-600">Error: ${error.message}</p>`;
    }
  });
});

document.getElementById("runEvaluation").addEventListener("click", async () => {
  const response = await fetch("/evaluate-model");
  const data = await response.json();

  const ul = document.createElement("ul");
  ul.className = "mt-4";

  const accuracy = `<li><strong>Accuracy:</strong> ${data.accuracy.toFixed(4)}</li>`;
  const loss = `<li><strong>Loss:</strong> ${data.loss.toFixed(4)}</li>`;
  const labels = Object.keys(data.report).filter(k => k !== 'accuracy' && k !== 'macro avg' && k !== 'weighted avg');

  const rows = labels.map(label => {
    const m = data.report[label];
    return `<tr>
      <td class="border px-2 py-1">${label}</td>
      <td class="border px-2 py-1">${(m.precision).toFixed(4)}</td>
      <td class="border px-2 py-1">${(m.recall).toFixed(4)}</td>
      <td class="border px-2 py-1">${(m["f1-score"]).toFixed(4)}</td>
    </tr>`;
  }).join("");

  const table = `
    <table class="table-auto mt-4 w-full border border-collapse border-gray-300 text-sm">
      <thead>
        <tr class="bg-gray-200">
          <th class="border px-2 py-1">Label</th>
          <th class="border px-2 py-1">Precision</th>
          <th class="border px-2 py-1">Recall</th>
          <th class="border px-2 py-1">F1-Score</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;

  const div = document.querySelector("#evaluate .mt-6");
  div.innerHTML += `<div class="mt-4">${accuracy}${loss}</div>${table}`;
});






