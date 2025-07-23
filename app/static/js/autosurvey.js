/**
 * AutoSurvey集成功能前端JavaScript
 */

class AutoSurveyApp {
    constructor() {
        this.selectedTasks = new Set();
        this.currentStep = 'task-selection';
        this.generationId = null;
        this.progressInterval = null;
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.loadTasks();
        this.showStep('task-selection');
    }
    
    bindEvents() {
        // 导航事件
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const step = e.target.closest('.nav-link').dataset.step;
                if (step) {
                    this.showStep(step);
                }
            });
        });
        
        // 任务选择相关事件
        document.getElementById('btn-refresh-tasks').addEventListener('click', () => this.loadTasks());
        document.getElementById('btn-select-all').addEventListener('click', () => this.selectAllTasks());
        document.getElementById('btn-clear-selection').addEventListener('click', () => this.clearSelection());
        document.getElementById('btn-apply-filters').addEventListener('click', () => this.applyFilters());
        
        // 步骤导航事件
        document.getElementById('btn-next-to-params').addEventListener('click', () => this.showStep('parameter-config'));
        document.getElementById('btn-back-to-tasks').addEventListener('click', () => this.showStep('task-selection'));
        document.getElementById('btn-start-generation').addEventListener('click', () => this.startGeneration());
        document.getElementById('btn-cancel-generation').addEventListener('click', () => this.cancelGeneration());
        document.getElementById('btn-new-generation').addEventListener('click', () => this.newGeneration());
        
        // 搜索事件
        document.getElementById('task-search').addEventListener('input', () => this.filterTasks());
        
        // 下载事件
        document.getElementById('btn-download-markdown').addEventListener('click', () => this.downloadResult('markdown'));
        document.getElementById('btn-download-pdf').addEventListener('click', () => this.downloadResult('pdf'));
        document.getElementById('btn-download-word').addEventListener('click', () => this.downloadResult('word'));
    }
    
    showStep(stepName) {
        // 隐藏所有步骤
        document.querySelectorAll('.step-content').forEach(step => {
            step.style.display = 'none';
        });
        
        // 显示目标步骤
        const targetStep = document.getElementById(`step-${stepName}`);
        if (targetStep) {
            targetStep.style.display = 'block';
        }
        
        // 更新导航状态
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        
        const activeLink = document.querySelector(`[data-step="${stepName}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }
        
        this.currentStep = stepName;
    }
    
    async loadTasks() {
        this.showLoading('正在加载任务列表...');
        
        try {
            const response = await fetch('/api/autosurvey/tasks');
            const data = await response.json();
            
            if (data.success) {
                this.renderTasks(data.tasks);
            } else {
                this.showError('加载任务失败: ' + data.message);
            }
        } catch (error) {
            this.showError('加载任务时出错: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }
    
    renderTasks(tasks) {
        const taskList = document.getElementById('task-list');
        taskList.innerHTML = '';
        
        if (tasks.length === 0) {
            taskList.innerHTML = `
                <div class="col-12">
                    <div class="alert alert-info text-center">
                        <i class="fas fa-info-circle"></i> 暂无可用任务
                    </div>
                </div>
            `;
            return;
        }
        
        tasks.forEach(task => {
            const taskCard = this.createTaskCard(task);
            taskList.appendChild(taskCard);
        });
    }
    
    createTaskCard(task) {
        const col = document.createElement('div');
        col.className = 'col-md-6 col-lg-4 mb-3';
        
        const isSelected = this.selectedTasks.has(task.task_id);
        
        col.innerHTML = `
            <div class="card task-card ${isSelected ? 'selected' : ''}" data-task-id="${task.task_id}">
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-start">
                        <h6 class="card-title">${task.task_name}</h6>
                        <div class="form-check">
                            <input class="form-check-input task-checkbox" type="checkbox" 
                                   ${isSelected ? 'checked' : ''} data-task-id="${task.task_id}">
                        </div>
                    </div>
                    
                    <p class="card-text text-muted small">
                        任务ID: ${task.task_id}
                    </p>
                    
                    <div class="task-stats">
                        <div class="stat-item">
                            <i class="fas fa-cubes text-primary"></i>
                            <span>${task.entity_count} 实体</span>
                        </div>
                        <div class="stat-item">
                            <i class="fas fa-link text-success"></i>
                            <span>${task.relation_count} 关系</span>
                        </div>
                    </div>
                    
                    <div class="mt-2">
                        <small class="text-muted">
                            <i class="fas fa-clock"></i> ${task.created_at}
                        </small>
                        <span class="badge bg-${this.getStatusColor(task.status)} ms-2">
                            ${this.getStatusText(task.status)}
                        </span>
                    </div>
                </div>
            </div>
        `;
        
        // 绑定点击事件
        const card = col.querySelector('.task-card');
        const checkbox = col.querySelector('.task-checkbox');
        
        card.addEventListener('click', (e) => {
            if (e.target.type !== 'checkbox') {
                checkbox.checked = !checkbox.checked;
                this.toggleTaskSelection(task.task_id, checkbox.checked);
            }
        });
        
        checkbox.addEventListener('change', (e) => {
            this.toggleTaskSelection(task.task_id, e.target.checked);
        });
        
        return col;
    }
    
    toggleTaskSelection(taskId, selected) {
        if (selected) {
            this.selectedTasks.add(taskId);
        } else {
            this.selectedTasks.delete(taskId);
        }
        
        // 更新卡片样式
        const card = document.querySelector(`[data-task-id="${taskId}"]`);
        if (card) {
            card.classList.toggle('selected', selected);
        }
        
        this.updateSelectionSummary();
        this.updateNextButton();
    }
    
    selectAllTasks() {
        document.querySelectorAll('.task-checkbox').forEach(checkbox => {
            checkbox.checked = true;
            this.selectedTasks.add(checkbox.dataset.taskId);
        });
        
        document.querySelectorAll('.task-card').forEach(card => {
            card.classList.add('selected');
        });
        
        this.updateSelectionSummary();
        this.updateNextButton();
    }
    
    clearSelection() {
        this.selectedTasks.clear();
        
        document.querySelectorAll('.task-checkbox').forEach(checkbox => {
            checkbox.checked = false;
        });
        
        document.querySelectorAll('.task-card').forEach(card => {
            card.classList.remove('selected');
        });
        
        this.updateSelectionSummary();
        this.updateNextButton();
    }
    
    updateSelectionSummary() {
        const summaryDiv = document.getElementById('selection-summary');
        const summaryContent = document.getElementById('summary-content');
        
        if (this.selectedTasks.size === 0) {
            summaryDiv.style.display = 'none';
            return;
        }
        
        summaryDiv.style.display = 'block';
        summaryContent.innerHTML = `
            已选择 <strong>${this.selectedTasks.size}</strong> 个任务
            <div class="mt-2">
                <small>任务ID: ${Array.from(this.selectedTasks).join(', ')}</small>
            </div>
        `;
    }
    
    updateNextButton() {
        const nextButton = document.getElementById('btn-next-to-params');
        nextButton.disabled = this.selectedTasks.size === 0;
    }
    
    filterTasks() {
        const searchTerm = document.getElementById('task-search').value.toLowerCase();
        const statusFilter = document.getElementById('status-filter').value;
        const entityCountFilter = document.getElementById('entity-count-filter').value;
        
        document.querySelectorAll('.task-card').forEach(card => {
            const taskName = card.querySelector('.card-title').textContent.toLowerCase();
            const taskId = card.dataset.taskId.toLowerCase();
            const status = card.querySelector('.badge').textContent.toLowerCase();
            const entityCount = parseInt(card.querySelector('.stat-item span').textContent);
            
            let visible = true;
            
            // 搜索过滤
            if (searchTerm && !taskName.includes(searchTerm) && !taskId.includes(searchTerm)) {
                visible = false;
            }
            
            // 状态过滤
            if (statusFilter && !status.includes(statusFilter)) {
                visible = false;
            }
            
            // 实体数量过滤
            if (entityCountFilter) {
                const [min, max] = this.parseEntityCountRange(entityCountFilter);
                if (entityCount < min || (max && entityCount > max)) {
                    visible = false;
                }
            }
            
            card.closest('.col-md-6').style.display = visible ? 'block' : 'none';
        });
    }
    
    parseEntityCountRange(range) {
        switch (range) {
            case '1-10': return [1, 10];
            case '11-50': return [11, 50];
            case '51+': return [51, null];
            default: return [0, null];
        }
    }
    
    async startGeneration() {
        if (this.selectedTasks.size === 0) {
            this.showError('请先选择任务');
            return;
        }
        
        const topic = document.getElementById('survey-topic').value.trim();
        if (!topic) {
            this.showError('请输入综述主题');
            return;
        }
        
        const params = this.collectGenerationParams();
        
        this.showStep('generation-progress');
        this.showLoading('正在启动综述生成...');
        
        try {
            const response = await fetch('/api/autosurvey/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    task_ids: Array.from(this.selectedTasks),
                    topic: topic,
                    parameters: params
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.generationId = data.generation_id;
                this.startProgressMonitoring();
            } else {
                this.showError('启动生成失败: ' + data.message);
                this.showStep('parameter-config');
            }
        } catch (error) {
            this.showError('启动生成时出错: ' + error.message);
            this.showStep('parameter-config');
        } finally {
            this.hideLoading();
        }
    }
    
    collectGenerationParams() {
        const outputFormats = [];
        if (document.getElementById('format-markdown').checked) outputFormats.push('markdown');
        if (document.getElementById('format-pdf').checked) outputFormats.push('pdf');
        if (document.getElementById('format-word').checked) outputFormats.push('word');
        
        return {
            section_num: parseInt(document.getElementById('section-num').value),
            subsection_len: parseInt(document.getElementById('subsection-len').value),
            rag_num: parseInt(document.getElementById('rag-num').value),
            outline_reference_num: parseInt(document.getElementById('outline-ref-num').value),
            model: document.getElementById('model-select').value,
            output_formats: outputFormats,
            include_lineage: document.getElementById('include-lineage').checked,
            include_visualization: document.getElementById('include-visualization').checked
        };
    }
    
    startProgressMonitoring() {
        this.progressInterval = setInterval(() => {
            this.checkProgress();
        }, 2000);
        
        this.checkProgress(); // 立即检查一次
    }
    
    async checkProgress() {
        if (!this.generationId) return;
        
        try {
            const response = await fetch(`/api/autosurvey/progress/${this.generationId}`);
            const data = await response.json();
            
            if (data.success) {
                this.updateProgress(data.progress);
                
                if (data.progress.status === 'completed') {
                    this.stopProgressMonitoring();
                    this.showGenerationResult(data.progress.result);
                } else if (data.progress.status === 'failed') {
                    this.stopProgressMonitoring();
                    this.showError('生成失败: ' + data.progress.error);
                }
            }
        } catch (error) {
            console.error('检查进度时出错:', error);
        }
    }
    
    updateProgress(progress) {
        document.getElementById('current-stage').textContent = progress.stage || '处理中...';
        document.getElementById('progress-percentage').textContent = `${Math.round(progress.percentage || 0)}%`;
        document.getElementById('progress-bar').style.width = `${progress.percentage || 0}%`;
        
        // 更新进度日志
        if (progress.logs) {
            const logContainer = document.getElementById('progress-log');
            logContainer.innerHTML = progress.logs.map(log => 
                `<div class="mb-1"><small class="text-muted">${log.timestamp}</small> ${log.message}</div>`
            ).join('');
            logContainer.scrollTop = logContainer.scrollHeight;
        }
    }
    
    stopProgressMonitoring() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
    }
    
    showGenerationResult(result) {
        this.showStep('result-display');
        
        // 更新结果摘要
        document.getElementById('result-word-count').textContent = result.word_count || '-';
        document.getElementById('result-section-count').textContent = result.section_count || '-';
        document.getElementById('result-reference-count').textContent = result.reference_count || '-';
        document.getElementById('result-quality-score').textContent = result.quality_score || '-';
        
        // 显示算法脉络分析
        if (result.algorithm_lineage) {
            document.getElementById('lineage-content').innerHTML = this.renderLineageAnalysis(result.algorithm_lineage);
        }
        
        // 显示综述内容预览
        document.getElementById('survey-content-preview').innerHTML = this.renderSurveyPreview(result.content);
    }
    
    renderLineageAnalysis(lineage) {
        if (!lineage.key_nodes || lineage.key_nodes.length === 0) {
            return '<p class="text-muted">暂无算法脉络分析数据</p>';
        }
        
        let html = '<div class="row">';
        
        // 关键节点
        html += '<div class="col-md-6">';
        html += '<h6>关键算法节点</h6>';
        html += '<ul class="list-group list-group-flush">';
        lineage.key_nodes.slice(0, 5).forEach(node => {
            html += `
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    <div>
                        <strong>${node.name}</strong>
                        <br><small class="text-muted">${node.year || '未知年份'}</small>
                    </div>
                    <span class="badge bg-primary rounded-pill">${node.importance_score?.toFixed(2) || '0.00'}</span>
                </li>
            `;
        });
        html += '</ul></div>';
        
        // 发展路径
        html += '<div class="col-md-6">';
        html += '<h6>主要发展路径</h6>';
        if (lineage.development_paths && lineage.development_paths.length > 0) {
            html += '<ul class="list-group list-group-flush">';
            lineage.development_paths.slice(0, 3).forEach((path, index) => {
                html += `
                    <li class="list-group-item">
                        <strong>路径 ${index + 1}</strong>
                        <br><small class="text-muted">长度: ${path.length} 节点</small>
                        ${path.start_year && path.end_year ? 
                            `<br><small class="text-muted">${path.start_year} - ${path.end_year}</small>` : ''}
                    </li>
                `;
            });
            html += '</ul>';
        } else {
            html += '<p class="text-muted">暂无发展路径数据</p>';
        }
        html += '</div>';
        
        html += '</div>';
        
        // 分析摘要
        if (lineage.analysis_summary) {
            html += `<div class="mt-3"><h6>分析摘要</h6><p>${lineage.analysis_summary}</p></div>`;
        }
        
        return html;
    }
    
    renderSurveyPreview(content) {
        if (!content) {
            return '<p class="text-muted">暂无内容预览</p>';
        }
        
        // 简单的Markdown渲染（实际项目中建议使用专业的Markdown解析器）
        return content
            .replace(/^# (.*$)/gim, '<h1>$1</h1>')
            .replace(/^## (.*$)/gim, '<h2>$1</h2>')
            .replace(/^### (.*$)/gim, '<h3>$1</h3>')
            .replace(/\*\*(.*)\*\*/gim, '<strong>$1</strong>')
            .replace(/\*(.*)\*/gim, '<em>$1</em>')
            .replace(/\n/gim, '<br>');
    }
    
    async downloadResult(format) {
        if (!this.generationId) {
            this.showError('没有可下载的结果');
            return;
        }
        
        try {
            const response = await fetch(`/api/autosurvey/download/${this.generationId}?format=${format}`);
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `survey_${this.generationId}.${format}`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            } else {
                this.showError('下载失败');
            }
        } catch (error) {
            this.showError('下载时出错: ' + error.message);
        }
    }
    
    newGeneration() {
        this.selectedTasks.clear();
        this.generationId = null;
        this.stopProgressMonitoring();
        this.showStep('task-selection');
        this.loadTasks();
    }
    
    cancelGeneration() {
        if (this.generationId) {
            fetch(`/api/autosurvey/cancel/${this.generationId}`, { method: 'POST' });
        }
        this.stopProgressMonitoring();
        this.showStep('parameter-config');
    }
    
    getStatusColor(status) {
        switch (status?.toLowerCase()) {
            case 'completed': return 'success';
            case 'processing': return 'primary';
            case 'failed': return 'danger';
            default: return 'secondary';
        }
    }
    
    getStatusText(status) {
        switch (status?.toLowerCase()) {
            case 'completed': return '已完成';
            case 'processing': return '处理中';
            case 'failed': return '失败';
            default: return '未知';
        }
    }
    
    showLoading(message) {
        document.getElementById('loading-message').textContent = message;
        const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
        modal.show();
    }
    
    hideLoading() {
        const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
        if (modal) {
            modal.hide();
        }
    }
    
    showError(message) {
        alert('错误: ' + message); // 简化的错误显示，实际项目中建议使用更好的UI组件
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new AutoSurveyApp();
});
