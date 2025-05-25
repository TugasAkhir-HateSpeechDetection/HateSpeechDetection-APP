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

