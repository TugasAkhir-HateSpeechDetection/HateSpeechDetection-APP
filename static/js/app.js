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


// Fungsi untuk menangani upload file
async function handleFileUpload() {
  const fileInput = document.getElementById('datasetFile');
  const uploadStatus = document.getElementById('uploadStatus');
  const file = fileInput.files[0];

  if (!file) {
    uploadStatus.classList.remove('hidden');
    uploadStatus.classList.remove('text-green-600', 'text-red-600');
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

    uploadStatus.classList.remove('hidden');
    uploadStatus.classList.remove('text-red-600', 'text-green-600');

    if (response.ok) {
      uploadStatus.classList.add('text-green-600');
      uploadStatus.innerText = `✅ ${result.message || 'File berhasil diupload!'}`;
    } else {
      uploadStatus.classList.add('text-red-600');
      uploadStatus.innerText = `❌ ${result.error || 'Upload gagal.'}`;
    }
  } catch (error) {
    uploadStatus.classList.remove('hidden');
    uploadStatus.classList.remove('text-green-600');
    uploadStatus.classList.add('text-red-600');
    uploadStatus.innerText = '❌ Terjadi kesalahan saat mengunggah file.';
    console.error(error);
  }
}

// Pasang event listener saat halaman sudah siap
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('uploadBtn').addEventListener('click', handleFileUpload);
});

async function fetchOriginalData() {
  const res = await fetch('/get_raw_data');
  const data = await res.json();
  renderTable('originalData', data, 'Data Awal');
}

async function runPreprocessing() {
  const res = await fetch('/preprocess', { method: 'POST' });
  const data = await res.json();
  renderTable('preprocessedData', data, 'Hasil Preprocessing');
}

function renderTable(containerId, rows, title) {
  const container = document.getElementById(containerId);
  if (!rows.length) {
    container.innerHTML = '<p>Data tidak ditemukan.</p>';
    return;
  }

  let table = `<h2 class="text-lg font-bold mb-2">${title}</h2>
               <table class="table-auto border border-collapse border-gray-400 w-full">
               <thead><tr>
                 <th class="border px-2 py-1 bg-gray-200">No.</th>
                 <th class="border px-2 py-1 bg-gray-200">Tweet</th>
               </tr></thead><tbody>`;

  rows.forEach((row, index) => {
    table += '<tr>';
    table += `<td class="border px-2 py-1 text-center">${index + 1}</td>`;
    table += `<td class="border px-2 py-1">${row.Tweet}</td>`;
    table += '</tr>';
  });

  table += '</tbody></table>';
  container.innerHTML = table;
}

document.getElementById('preprocessBtn').addEventListener('click', runPreprocessing);
fetchOriginalData();

// document.addEventListener('DOMContentLoaded', () => {
//   const tokenizeBtn = document.getElementById('tokenizeBtn');
//   if (tokenizeBtn) {
//     tokenizeBtn.addEventListener('click', async () => {
//       const resultDiv = document.getElementById('tokenResult');
//       resultDiv.innerHTML = 'Memproses tokenisasi...';

//       try {
//         const response = await fetch('/run_tokenization', { method: 'POST' });
//         const data = await response.json();

//         if (data.error) {
//           resultDiv.innerHTML = `<p class="text-red-600">Terjadi kesalahan: ${data.error}</p>`;
//           return;
//         }

//         let tableHTML = `
//           <table class="min-w-full table-auto border-collapse border border-gray-300">
//             <thead>
//               <tr class="bg-gray-200">
//                 <th class="border border-gray-300 px-4 py-2">Tweet</th>
//                 <th class="border border-gray-300 px-4 py-2">Tokens</th>
//                 <th class="border border-gray-300 px-4 py-2">Token IDs</th>
//               </tr>
//             </thead>
//             <tbody>
//         `;

//         data.forEach(row => {
//           const tokens = row.Tokens.join(', ');
//           const ids = row.Token_IDs.join(', ');
//           tableHTML += `
//             <tr>
//               <td class="border px-2 py-1 text-center">${index + 1}</td>
//               <td class="border border-gray-300 px-4 py-2">${row.Tweet}</td>
//               <td class="border border-gray-300 px-4 py-2">${tokens}</td>
//               <td class="border border-gray-300 px-4 py-2">${ids}</td>
//             </tr>
//           `;
//         });

//         tableHTML += `</tbody></table>`;
//         resultDiv.innerHTML = tableHTML;

//       } catch (error) {
//         resultDiv.innerHTML = `<p class="text-red-600">Gagal memproses: ${error.message}</p>`;
//       }
//     });
//   }
// });

// document.getElementById('tokenizeBtn').addEventListener('click', function () {
//   fetch('/run_tokenization', {
//     method: 'POST'
//   })
//     .then(response => response.json())
//     .then(data => {
//       if (data.error) {
//         alert("Terjadi kesalahan: " + data.error);
//         return;
//       }

//       // --- Tampilkan Pre Data ---
//       const preData = data.pre_data;
//       const preDisplay = document.querySelector('#tokenize pre');
//       preDisplay.textContent = preData.map(row => `${row.No}. ${row.Tweet}`).join('\n');

//       // --- Tampilkan Tokenization Result dalam bentuk tabel ---
//       const tokenData = data.token_data;
//       const resultDiv = document.getElementById('tokenResult');

//       const table = document.createElement('table');
//       table.className = 'min-w-full border border-gray-300 mt-2 text-sm';

//       const thead = document.createElement('thead');
//       thead.innerHTML = `
//         <tr class="bg-gray-200">
//           <th class="border px-3 py-2 text-left">No</th>
//           <th class="border px-3 py-2 text-left">Tweet</th>
//           <th class="border px-3 py-2 text-left">Tokens</th>
//           <th class="border px-3 py-2 text-left">Token IDs</th>
//         </tr>
//       `;
//       table.appendChild(thead);

//       const tbody = document.createElement('tbody');

//       tokenData.forEach(row => {
//         const tr = document.createElement('tr');
//         tr.innerHTML = `
//           <td class="border px-3 py-1 align-top">${row.No}</td>
//           <td class="border px-3 py-1 align-top">${row.Tweet}</td>
//           <td class="border px-3 py-1 align-top">${Array.isArray(row.Tokens) ? row.Tokens.join(', ') : row.Tokens}</td>
//           <td class="border px-3 py-1 align-top">${Array.isArray(row.Token_IDs) ? row.Token_IDs.join(', ') : row.Token_IDs}</td>
//         `;
//         tbody.appendChild(tr);
//       });

//       table.appendChild(tbody);
//       resultDiv.innerHTML = ''; // Bersihkan sebelum render baru
//       resultDiv.appendChild(table);
//     })
//     .catch(error => {
//       alert("Gagal mengambil data tokenisasi: " + error);
//     });
// });

// Fungsi untuk load preprocessed text saat halaman dimuat
function loadPreprocessedText() {
  fetch('/get_preprocessed')
    .then(response => response.json())
    .then(data => {
      const preDisplay = document.getElementById('preTokenText');
      if (data.error) {
        preDisplay.textContent = "Data belum tersedia.";
      } else {
        const texts = data.pre_data;
        preDisplay.textContent = texts.map(row => `${row.No}. ${row.Tweet}`).join('\n');
      }
    })
    .catch(err => {
      document.getElementById('preTokenText').textContent = "Gagal mengambil data.";
    });
}

// Panggil saat halaman dimuat
document.addEventListener('DOMContentLoaded', loadPreprocessedText);

// Fungsi untuk tokenisasi
document.getElementById('tokenizeBtn').addEventListener('click', function () {
  fetch('/run_tokenization', {
    method: 'POST'
  })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        alert("Terjadi kesalahan: " + data.error);
        return;
      }

      const tokenData = data.token_data;
      const resultDiv = document.getElementById('tokenResult');

      const table = document.createElement('table');
      table.className = 'min-w-full border border-gray-300 mt-2 text-sm';

      const thead = document.createElement('thead');
      thead.innerHTML = `
        <tr class="bg-gray-200">
          <th class="border px-3 py-2 text-left">No</th>
          <th class="border px-3 py-2 text-left">Tweet</th>
          <th class="border px-3 py-2 text-left">Tokens</th>
          <th class="border px-3 py-2 text-left">Token IDs</th>
        </tr>
      `;
      table.appendChild(thead);

      const tbody = document.createElement('tbody');

      tokenData.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td class="border px-3 py-1 align-top">${row.No}</td>
          <td class="border px-3 py-1 align-top">${row.Tweet}</td>
          <td class="border px-3 py-1 align-top">${Array.isArray(row.Tokens) ? row.Tokens.join(', ') : row.Tokens}</td>
          <td class="border px-3 py-1 align-top">${Array.isArray(row.Token_IDs) ? row.Token_IDs.join(', ') : row.Token_IDs}</td>
        `;
        tbody.appendChild(tr);
      });

      table.appendChild(tbody);
      resultDiv.innerHTML = '';
      resultDiv.appendChild(table);
    })
    .catch(error => {
      alert("Gagal mengambil data tokenisasi: " + error);
    });
});

document.getElementById("btnTune").addEventListener("click", function () {
  fetch("/tune", {
    method: "POST",
  })
    .then((response) => response.json())
    .then((data) => {
      const tableBody = document.querySelector("#tune table tbody");
      tableBody.innerHTML = "";

      data.results.forEach((item) => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>${item.iteration}</td>
          <td>${item.params.epochs}</td>
          <td>${item.params.units}</td>
          <td>${item.params.learning_rate}</td>
          <td>${item.params.batch_size}</td>
        `;
        tableBody.appendChild(row);
      });
    })
    .catch((error) => {
      alert("Terjadi kesalahan saat proses tuning.");
      console.error(error);
    });
});
document.getElementById('trainBtn').addEventListener('click', () => {
  const logArea = document.getElementById('logArea');
  logArea.textContent = 'Training dimulai...\n';

  const eventSource = new EventSource('/train');

  eventSource.onmessage = (event) => {
    if (event.data === '[TRAINING_COMPLETED]') {
      logArea.textContent += '\n[✓] Training selesai.\n';
      eventSource.close();
    } else {
      logArea.textContent += event.data + '\n';
      logArea.scrollTop = logArea.scrollHeight;
    }
  };

  eventSource.onerror = (err) => {
    logArea.textContent += '\n[!] Terjadi error selama training.\n';
    eventSource.close();
  };
});

document.addEventListener('DOMContentLoaded', () => {
  fetch('/best-params')
    .then(res => res.json())
    .then(data => {
      document.getElementById('lrCell').textContent = data.learning_rate;
      document.getElementById('bsCell').textContent = data.batch_size;
      document.getElementById('epochCell').textContent = data.epochs;
      document.getElementById('unitsCell').textContent = data.units;
    })
    .catch(err => {
      console.error('Gagal memuat hyperparameter:', err);
    });
});

document.querySelector('#test button').addEventListener('click', () => {
  const textarea = document.querySelector('#test textarea');
  const resultList = document.querySelector('#test ul');
  const inputText = textarea.value;

  fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: inputText})
  })
  .then(res => res.json())
  .then(data => {
    resultList.innerHTML = '';
    if (data.labels.length === 0) {
      resultList.innerHTML = '<li class="text-gray-600">Tidak terdeteksi ujaran kebencian.</li>';
    } else {
      data.labels.forEach(label => {
        const li = document.createElement('li');
        li.textContent = label;
        resultList.appendChild(li);
      });
    }
  })
  .catch(err => {
    resultList.innerHTML = '<li class="text-red-600">Terjadi kesalahan saat prediksi.</li>';
    console.error(err);
  });
});


