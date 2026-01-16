// Dashboard State
let currentJobId = null;
let jobRefreshInterval = null;
let logEventSource = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    loadModels();
    loadAvailableModels();
    loadJobs();
    loadSystemInfo();
    startStatsRefresh();
    setupFilters();
});

// ==================== TAB MANAGEMENT ====================

function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;
            switchTab(tabName);
        });
    });
}

function switchTab(tabName) {
    // Update buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });

    // Update content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `${tabName}-tab`);
    });

    // Load data for the tab
    if (tabName === 'models') {
        loadModels();
    } else if (tabName === 'jobs') {
        loadJobs();
    } else if (tabName === 'system') {
        loadSystemInfo();
    }
}

function initDatasetTabs() {
    const datasetBtns = document.querySelectorAll('.dataset-tab-btn');
    datasetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const source = btn.dataset.source;
            
            // Update buttons
            datasetBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update content
            document.querySelectorAll('.dataset-content').forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(`${source}-content`).classList.add('active');
        });
    });
}

// ==================== MODEL LIBRARY ====================

async function loadModels() {
    const grid = document.getElementById('models-grid');
    grid.innerHTML = '<div class="loading-spinner"></div>';

    try {
        const response = await fetch('/api/dashboard/models');
        const data = await response.json();

        if (data.models.length === 0) {
            grid.innerHTML = '<div class="empty-state">No models found</div>';
            return;
        }

        grid.innerHTML = data.models.map(model => `
            <div class="model-card" data-category="${model.category}">
                <div class="model-header">
                    <div class="model-icon">${getModelIcon(model.category)}</div>
                    <span class="model-format">${model.format}</span>
                </div>
                <div class="model-body">
                    <h4 class="model-name">${escapeHtml(model.name)}</h4>
                    <div class="model-meta">
                        <span class="category-badge">${model.category}</span>
                        <span class="model-size">${model.size_mb} MB</span>
                    </div>
                    <div class="model-path">${escapeHtml(model.path)}</div>
                </div>
                <div class="model-actions">
                    <button class="btn-icon" onclick="downloadModel('${model.category}', '${escapeHtml(model.filename)}')" title="Download">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/>
                        </svg>
                    </button>

                </div>
            </div>
        `).join('');

    } catch (error) {
        grid.innerHTML = `<div class="error-state">Failed to load models: ${error.message}</div>`;
    }
}

function getModelIcon(category) {
    const icons = {
        vehicle: '🚗',
        plate: '🔖',
        ocr: '📝',
        tracking: '📍',
        pretrained: '⚙️'
    };
    return icons[category] || '📦';
}

async function downloadModel(category, filename) {
    try {
        const url = `/api/dashboard/models/download/${encodeURIComponent(category)}/${encodeURIComponent(filename)}`;
        window.location.href = url;
    } catch (error) {
        alert(`Failed to download model: ${error.message}`);
    }
}

function refreshModels() {
    loadModels();
}

// ==================== TRAINING ====================

async function loadAvailableModels() {
    try {
        const response = await fetch('/api/dashboard/training/models');
        const data = await response.json();

        const select = document.getElementById('model-type');
        select.innerHTML = '<option value="">Select model...</option>' +
            data.models.map(m => `<option value="${m.value}">${m.name}</option>`).join('');
    } catch (error) {
        console.error('Failed to load models:', error);
    }
}

async function startTraining() {
    // Collect form data
    const jobName = document.getElementById('job-name').value.trim();
    const modelType = document.getElementById('model-type').value;
    const epochs = parseInt(document.getElementById('epochs').value);
    const batchSize = parseInt(document.getElementById('batch-size').value);
    const imageSize = parseInt(document.getElementById('image-size').value);
    const device = document.getElementById('device').value;
    const workers = parseInt(document.getElementById('workers').value);

    // Validate basic fields
    if (!jobName) {
        alert('Please enter a job name');
        return;
    }
    if (!modelType) {
        alert('Please select a model architecture');
        return;
    }

    // Collect Roboflow dataset info
    const apiKey = document.getElementById('rf-api-key').value.trim();
    const workspace = document.getElementById('rf-workspace').value.trim();
    const project = document.getElementById('rf-project').value.trim();
    const version = parseInt(document.getElementById('rf-version').value);

    if (!apiKey || !workspace || !project || !version) {
        alert('Please fill in all Roboflow fields');
        return;
    }

    // Build request
    const config = {
        job_name: jobName,
        model_type: modelType,
        epochs: epochs,
        batch_size: batchSize,
        image_size: imageSize,
        device: device,
        workers: workers,
        roboflow_dataset: {
            api_key: apiKey,
            workspace: workspace,
            project: project,
            version: version
        }
    };

    try {
        const response = await fetch('/api/dashboard/training/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'Failed to start training');
        }

        alert(`Training started successfully! Job ID: ${result.job_id}`);
        
        // Switch to jobs tab
        switchTab('jobs');
        
        // Open job modal
        setTimeout(() => {
            openJobModal(result.job_id);
        }, 500);

    } catch (error) {
        alert(`Failed to start training: ${error.message}`);
    }
}

function resetTrainingForm() {
    document.getElementById('job-name').value = '';
    document.getElementById('model-type').value = '';
    document.getElementById('epochs').value = 100;
    document.getElementById('batch-size').value = 16;
    document.getElementById('image-size').value = 640;
    document.getElementById('device').value = '0';
    document.getElementById('workers').value = 8;
    document.getElementById('rf-api-key').value = '';
    document.getElementById('rf-workspace').value = '';
    document.getElementById('rf-project').value = '';
    document.getElementById('rf-version').value = '';
}

// ==================== TRAINING JOBS ====================

async function loadJobs() {
    const list = document.getElementById('jobs-list');
    list.innerHTML = '<div class="loading-spinner"></div>';

    try {
        const response = await fetch('/api/dashboard/training/jobs');
        const data = await response.json();

        if (data.jobs.length === 0) {
            list.innerHTML = '<div class="empty-state">No training jobs yet</div>';
            return;
        }

        list.innerHTML = data.jobs.map(job => `
            <div class="job-card ${job.status}" onclick="openJobModal('${job.id}')">
                <div class="job-header">
                    <h4>${escapeHtml(job.name)}</h4>
                    <span class="status-badge status-${job.status}">${job.status}</span>
                </div>
                <div class="job-meta">
                    <span>Model: ${job.config.model_type}</span>
                    <span>Epochs: ${job.config.epochs}</span>
                    <span>Batch: ${job.config.batch_size}</span>
                </div>
                <div class="job-progress">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${job.progress}%"></div>
                    </div>
                    <span class="progress-text">${job.progress}%</span>
                </div>
                <div class="job-footer">
                    <span class="job-time">${formatDate(job.started_at)}</span>
                    <span class="job-message">${escapeHtml(job.message)}</span>
                </div>
            </div>
        `).join('');

    } catch (error) {
        list.innerHTML = `<div class="error-state">Failed to load jobs: ${error.message}</div>`;
    }
}

function refreshJobs() {
    loadJobs();
}

async function openJobModal(jobId) {
    currentJobId = jobId;
    
    const modal = document.getElementById('job-modal');
    modal.style.display = 'flex';

    // Start refresh interval
    refreshJobDetails();
    jobRefreshInterval = setInterval(refreshJobDetails, 2000);

    // Start streaming logs
    streamJobLogs(jobId);
}

async function refreshJobDetails() {
    if (!currentJobId) return;

    try {
        const response = await fetch(`/api/dashboard/training/jobs/${currentJobId}`);
        const job = await response.json();

        document.getElementById('modal-job-name').textContent = job.name;
        document.getElementById('modal-job-status').textContent = job.status;
        document.getElementById('modal-job-status').className = `status-badge status-${job.status}`;
        document.getElementById('modal-job-progress').textContent = `${job.progress}%`;
        document.getElementById('modal-progress-fill').style.width = `${job.progress}%`;
        document.getElementById('modal-epoch').textContent = job.epoch;
        document.getElementById('modal-total-epochs').textContent = job.total_epochs;

        // Update buttons based on status
        const cancelBtn = document.getElementById('cancel-job-btn');
        const closeBtn = document.getElementById('close-modal-btn');
        
        if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
            cancelBtn.style.display = 'none';
            closeBtn.style.display = 'inline-flex';
            
            // Stop refresh
            if (jobRefreshInterval) {
                clearInterval(jobRefreshInterval);
                jobRefreshInterval = null;
            }
        }

    } catch (error) {
        console.error('Failed to refresh job details:', error);
    }
}

function streamJobLogs(jobId) {
    // Close existing stream
    if (logEventSource) {
        logEventSource.close();
    }

    const logsContainer = document.getElementById('modal-logs');
    logsContainer.innerHTML = '';

    logEventSource = new EventSource(`/api/dashboard/training/logs/${jobId}`);

    logEventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.line) {
            const logLine = document.createElement('div');
            logLine.className = 'log-line';
            logLine.textContent = data.line;
            logsContainer.appendChild(logLine);
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }

        if (data.status) {
            logEventSource.close();
            logEventSource = null;
        }
    };

    logEventSource.onerror = () => {
        console.error('Log stream error');
        logEventSource.close();
        logEventSource = null;
    };
}

function closeJobModal() {
    const modal = document.getElementById('job-modal');
    modal.style.display = 'none';
    currentJobId = null;

    if (jobRefreshInterval) {
        clearInterval(jobRefreshInterval);
        jobRefreshInterval = null;
    }

    if (logEventSource) {
        logEventSource.close();
        logEventSource = null;
    }

    // Refresh jobs list
    loadJobs();
}

async function cancelJob() {
    if (!currentJobId) return;
    
    if (!confirm('Are you sure you want to cancel this training job?')) {
        return;
    }

    try {
        const response = await fetch(`/api/dashboard/training/jobs/${currentJobId}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error('Failed to cancel job');
        }

        alert('Job cancelled');
        refreshJobDetails();
    } catch (error) {
        alert(`Failed to cancel job: ${error.message}`);
    }
}

function downloadJobLogs() {
    if (!currentJobId) return;
    
    const logsContainer = document.getElementById('modal-logs');
    const logs = Array.from(logsContainer.querySelectorAll('.log-line'))
        .map(line => line.textContent)
        .join('\n');
    
    const blob = new Blob([logs], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentJobId}_logs.txt`;
    a.click();
    URL.revokeObjectURL(url);
}

// ==================== SYSTEM INFO ====================

async function loadSystemInfo() {
    loadGPUInfo();
    loadStorageInfo();
    loadMetricsInfo();
}

async function loadGPUInfo() {
    const container = document.getElementById('gpu-info');
    container.innerHTML = '<div class="loading-spinner"></div>';

    try {
        const response = await fetch('/api/dashboard/system/gpu');
        const data = await response.json();

        if (!data.available) {
            container.innerHTML = '<div class="info-item">No GPU available or PyTorch not installed</div>';
            return;
        }

        container.innerHTML = `
            <div class="info-item">
                <span class="info-label">Available:</span>
                <span class="info-value">Yes (${data.device_count} device${data.device_count > 1 ? 's' : ''})</span>
            </div>
            <div class="info-item">
                <span class="info-label">CUDA Version:</span>
                <span class="info-value">${data.cuda_version || 'N/A'}</span>
            </div>
            ${data.gpus.map((gpu, i) => `
                <div class="gpu-card">
                    <h4>GPU ${gpu.id}: ${escapeHtml(gpu.name)}</h4>
                    <div class="info-item">
                        <span class="info-label">Compute Capability:</span>
                        <span class="info-value">${gpu.compute_capability}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Total Memory:</span>
                        <span class="info-value">${gpu.total_memory_gb} GB</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Free Memory:</span>
                        <span class="info-value">${gpu.free_memory_gb} GB</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Utilization:</span>
                        <span class="info-value">${gpu.utilization}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${gpu.utilization}%"></div>
                    </div>
                </div>
            `).join('')}
        `;
    } catch (error) {
        container.innerHTML = `<div class="error-state">Failed to load GPU info: ${error.message}</div>`;
    }
}

async function loadStorageInfo() {
    const container = document.getElementById('storage-info');
    container.innerHTML = '<div class="loading-spinner"></div>';

    try {
        const response = await fetch('/api/dashboard/system/storage');
        const data = await response.json();

        container.innerHTML = data.storage.map(storage => `
            <div class="storage-card">
                <h4>${storage.location}</h4>
                <div class="info-item">
                    <span class="info-label">Path:</span>
                    <span class="info-value small">${escapeHtml(storage.path)}</span>
                </div>
                ${storage.total_gb ? `
                    <div class="info-item">
                        <span class="info-label">Total:</span>
                        <span class="info-value">${storage.total_gb} GB</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Used:</span>
                        <span class="info-value">${storage.used_gb} GB</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Free:</span>
                        <span class="info-value">${storage.free_gb} GB</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${storage.usage_percent}%"></div>
                    </div>
                ` : `
                    <div class="info-item">
                        <span class="info-label">Size:</span>
                        <span class="info-value">${storage.size_gb} GB</span>
                    </div>
                `}
            </div>
        `).join('');
    } catch (error) {
        container.innerHTML = `<div class="error-state">Failed to load storage info: ${error.message}</div>`;
    }
}

async function loadMetricsInfo() {
    const container = document.getElementById('metrics-info');
    container.innerHTML = '<div class="loading-spinner"></div>';

    try {
        const response = await fetch('/api/dashboard/system/metrics');
        const data = await response.json();

        container.innerHTML = `
            <div class="metric-card">
                <h4>CPU</h4>
                <div class="info-item">
                    <span class="info-label">Usage:</span>
                    <span class="info-value">${data.cpu.usage_percent}%</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Cores:</span>
                    <span class="info-value">${data.cpu.count} (${data.cpu.count_physical} physical)</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${data.cpu.usage_percent}%"></div>
                </div>
            </div>

            <div class="metric-card">
                <h4>Memory</h4>
                <div class="info-item">
                    <span class="info-label">Usage:</span>
                    <span class="info-value">${data.memory.usage_percent}%</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Total:</span>
                    <span class="info-value">${data.memory.total_gb} GB</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Available:</span>
                    <span class="info-value">${data.memory.available_gb} GB</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${data.memory.usage_percent}%"></div>
                </div>
            </div>
        `;
    } catch (error) {
        container.innerHTML = `<div class="error-state">Failed to load metrics: ${error.message}</div>`;
    }
}

function refreshSystemInfo() {
    loadSystemInfo();
}

// ==================== STATS BAR ====================

async function updateStatsBar() {
    try {
        // GPU Status
        const gpuResponse = await fetch('/api/dashboard/system/gpu');
        const gpuData = await gpuResponse.json();
        const gpuStatus = document.getElementById('gpu-status');
        
        if (gpuData.available && gpuData.gpus.length > 0) {
            const gpu = gpuData.gpus[0];
            gpuStatus.textContent = `${gpu.name.split(' ')[0]} ${gpu.utilization}%`;
        } else {
            gpuStatus.textContent = 'N/A';
        }

        // Metrics
        const metricsResponse = await fetch('/api/dashboard/system/metrics');
        const metricsData = await metricsResponse.json();
        
        document.getElementById('cpu-usage').textContent = `${metricsData.cpu.usage_percent}%`;
        document.getElementById('ram-usage').textContent = `${metricsData.memory.usage_percent}%`;

        // Storage
        const storageResponse = await fetch('/api/dashboard/system/storage');
        const storageData = await storageResponse.json();
        const workspace = storageData.storage.find(s => s.location === 'Workspace');
        if (workspace) {
            document.getElementById('storage-free').textContent = `${workspace.free_gb} GB Free`;
        }
    } catch (error) {
        console.error('Failed to update stats bar:', error);
    }
}

function startStatsRefresh() {
    updateStatsBar();
    setInterval(updateStatsBar, 5000); // Update every 5 seconds
}

// ==================== FILTERS ====================

function setupFilters() {
    const searchInput = document.getElementById('model-search');
    const categoryFilter = document.getElementById('category-filter');

    searchInput.addEventListener('input', filterModels);
    categoryFilter.addEventListener('change', filterModels);
}

function filterModels() {
    const searchTerm = document.getElementById('model-search').value.toLowerCase();
    const category = document.getElementById('category-filter').value;
    
    const cards = document.querySelectorAll('.model-card');
    
    cards.forEach(card => {
        const name = card.querySelector('.model-name').textContent.toLowerCase();
        const cardCategory = card.dataset.category;
        
        const matchesSearch = name.includes(searchTerm);
        const matchesCategory = category === 'all' || cardCategory === category;
        
        card.style.display = matchesSearch && matchesCategory ? 'block' : 'none';
    });
}

// ==================== UTILITIES ====================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatDate(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString();
}

// Close modal on backdrop click
document.getElementById('job-modal')?.addEventListener('click', (e) => {
    if (e.target.id === 'job-modal') {
        closeJobModal();
    }
});
