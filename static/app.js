(() => {
  const root = document.documentElement;
  const setTheme = (theme) => {
    root.dataset.theme = theme;
    localStorage.setItem('dermo-theme', theme);
    document.querySelectorAll('[data-theme-toggle]').forEach((button) => {
      button.setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
      button.dataset.themeState = theme;
    });
  };

  setTheme(root.dataset.theme || 'light');

  document.querySelectorAll('[data-theme-toggle]').forEach((button) => {
    button.addEventListener('click', () => {
      setTheme(root.dataset.theme === 'dark' ? 'light' : 'dark');
    });
  });

  const uploadInput = document.getElementById('image-input');
  const preview = document.getElementById('upload-preview');
  const dropzone = document.querySelector('[data-dropzone]');

  const renderPreview = (file) => {
    if (!file || !preview) return;
    const url = URL.createObjectURL(file);
    preview.classList.add('has-image');
    preview.innerHTML = `
      <img src="${url}" alt="Uploaded lesion preview">
      <span class="upload-state">Image ready for AI analysis</span>
      <strong>${file.name}</strong>
      <small>Preview loaded. Use a clear, well-lit crop for best results.</small>
    `;
  };

  if (uploadInput) {
    uploadInput.addEventListener('change', () => renderPreview(uploadInput.files[0]));
  }

  if (dropzone && uploadInput) {
    ['dragenter', 'dragover'].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add('is-dragging');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove('is-dragging');
      });
    });

    dropzone.addEventListener('drop', (event) => {
      const files = event.dataTransfer.files;
      if (!files.length) return;
      uploadInput.files = files;
      renderPreview(files[0]);
    });
  }

  document.querySelectorAll('[data-loading-form]').forEach((form) => {
    form.addEventListener('submit', () => {
      const button = form.querySelector('[data-loading-button]');
      if (!button) return;
      button.classList.add('is-loading');
      button.disabled = true;
      button.textContent = button.dataset.loadingLabel || 'Processing...';
    });
  });

  const background = document.querySelector('.medical-background');
  if (background && window.matchMedia('(prefers-reduced-motion: no-preference)').matches) {
    window.addEventListener('pointermove', (event) => {
      const x = (event.clientX / window.innerWidth - 0.5) * 14;
      const y = (event.clientY / window.innerHeight - 0.5) * 14;
      background.style.setProperty('--parallax-x', `${x}px`);
      background.style.setProperty('--parallax-y', `${y}px`);
    }, { passive: true });
  }
})();
