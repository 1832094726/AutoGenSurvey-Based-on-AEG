// 论文分析系统JavaScript
class PaperAnalysisSystem {
    constructor() {
        this.init();
    }

    init() {
        this.loadTasks();
        this.setupFileUploads();
        this.setupForms();
        this.loadAnalysisHistory();
        this.setupHistoryRefresh();
    }

    // 加载任务列表
    async loadTasks() {
        try {
            const response = await fetch('/api/analysis/tasks');
            const data = await response.json();

            if (data.success) {
                this.populateTaskSelects(data.tasks);

                // 显示任务加载状态信息
                if (data.message) {
                    if (data.tasks.length === 0 || (data.tasks.length > 0 && data.tasks[0].status === 'example')) {
                        this.showInfo(data.message);
                    } else {
                        console.log('任务加载成功:', data.message);
                    }
                }
            } else {
                console.error('加载任务失败:', data.message);
                this.showError('加载任务列表失败: ' + data.message);
                // 即使失败也尝试填充任务选择框（可能有示例数据）
                if (data.tasks) {
                    this.populateTaskSelects(data.tasks);
                }
            }
        } catch (error) {
            console.error('加载任务出错:', error);
            this.showError('网络错误，无法加载任务列表');
            // 提供备用的示例数据
            this.populateTaskSelects([
                {
                    task_id: 'network_error',
                    name: '网络连接失败',
                    description: '无法连接到服务器',
                    status: 'error'
                }
            ]);
        }
    }

    // 填充任务选择框
    populateTaskSelects(tasks) {
        const taskSelects = ['taskSelect', 'taskSelect2'];

        taskSelects.forEach(selectId => {
            const select = document.getElementById(selectId);
            select.innerHTML = '<option value="">请选择任务</option>';

            if (tasks && tasks.length > 0) {
                tasks.forEach(task => {
                    const option = document.createElement('option');
                    option.value = task.task_id || task.id;

                    // 构建更详细的显示文本
                    let displayText = task.name || task.task_id || task.id;

                    // 添加实体和关系数量信息
                    if (task.entity_count > 0 || task.relation_count > 0) {
                        displayText += ` (${task.entity_count}个实体, ${task.relation_count}个关系)`;
                    }

                    // 如果是示例任务，添加提示
                    if (task.status === 'example' || task.status === 'error') {
                        displayText += ' - 示例数据';
                        option.disabled = true;
                        option.style.color = '#6c757d';
                    }

                    option.textContent = displayText;
                    option.title = task.description || ''; // 添加悬停提示
                    select.appendChild(option);
                });

                // 如果只有示例数据，添加提示信息
                if (tasks.length > 0 && tasks[0].status === 'example') {
                    const infoOption = document.createElement('option');
                    infoOption.value = '';
                    infoOption.textContent = '--- 请先上传论文生成任务数据 ---';
                    infoOption.disabled = true;
                    infoOption.style.fontStyle = 'italic';
                    infoOption.style.color = '#dc3545';
                    select.appendChild(infoOption);
                }
            } else {
                const noTaskOption = document.createElement('option');
                noTaskOption.value = '';
                noTaskOption.textContent = '暂无可用任务，请先上传论文';
                noTaskOption.disabled = true;
                select.appendChild(noTaskOption);
            }
        });
    }

    // 设置文件上传
    setupFileUploads() {
        // PDF文件上传区域
        const uploadAreas = [
            { area: 'paperUpload1', input: 'paperFile1', display: 'fileName1' },
            { area: 'paperUpload2', input: 'paperFile2', display: 'fileName2' },
            { area: 'paperUpload3', input: 'paperFile3', display: 'fileName3' }
        ];

        uploadAreas.forEach(upload => {
            const area = document.getElementById(upload.area);
            const input = document.getElementById(upload.input);
            const display = document.getElementById(upload.display);

            if (!area || !input || !display) return;

            // 点击上传
            area.addEventListener('click', () => input.click());

            // 文件选择
            input.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) {
                    this.handleFileSelect(file, display, 'pdf');
                }
            });

            // 拖拽上传
            area.addEventListener('dragover', (e) => {
                e.preventDefault();
                area.classList.add('dragover');
            });

            area.addEventListener('dragleave', () => {
                area.classList.remove('dragover');
            });

            area.addEventListener('drop', (e) => {
                e.preventDefault();
                area.classList.remove('dragover');

                const file = e.dataTransfer.files[0];
                if (file && file.type === 'application/pdf') {
                    input.files = e.dataTransfer.files;
                    this.handleFileSelect(file, display, 'pdf');
                } else {
                    this.showError('请上传PDF文件');
                }
            });
        });

        // 提取结果文件上传区域
        const extractedUploadAreas = [
            { area: 'extractedUpload1', input: 'extractedFile1', display: 'extractedFileName1' },
            { area: 'extractedUpload2', input: 'extractedFile2', display: 'extractedFileName2' }
        ];

        extractedUploadAreas.forEach(upload => {
            const area = document.getElementById(upload.area);
            const input = document.getElementById(upload.input);
            const display = document.getElementById(upload.display);

            if (!area || !input || !display) return;

            // 点击上传
            area.addEventListener('click', () => input.click());

            // 文件选择
            input.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) {
                    this.handleFileSelect(file, display, 'json');
                }
            });

            // 拖拽上传
            area.addEventListener('dragover', (e) => {
                e.preventDefault();
                area.classList.add('dragover');
            });

            area.addEventListener('dragleave', () => {
                area.classList.remove('dragover');
            });

            area.addEventListener('drop', (e) => {
                e.preventDefault();
                area.classList.remove('dragover');

                const file = e.dataTransfer.files[0];
                if (file && file.type === 'application/json') {
                    input.files = e.dataTransfer.files;
                    this.handleFileSelect(file, display, 'json');
                } else {
                    this.showError('请上传JSON文件');
                }
            });
        });
    }

    // 处理文件选择
    handleFileSelect(file, displayElement, fileType = 'pdf') {
        if (fileType === 'pdf') {
            if (file.type !== 'application/pdf') {
                this.showError('请选择PDF文件');
                return;
            }

            if (file.size > 100 * 1024 * 1024) { // 100MB
                this.showError('文件大小不能超过100MB');
                return;
            }

            displayElement.innerHTML = `
                <i class="fas fa-file-pdf text-danger"></i>
                ${file.name} (${this.formatFileSize(file.size)})
            `;
        } else if (fileType === 'json') {
            if (file.type !== 'application/json' && !file.name.endsWith('.json')) {
                this.showError('请选择JSON文件');
                return;
            }

            if (file.size > 10 * 1024 * 1024) { // 10MB
                this.showError('JSON文件大小不能超过10MB');
                return;
            }

            displayElement.innerHTML = `
                <i class="fas fa-file-code text-info"></i>
                ${file.name} (${this.formatFileSize(file.size)})
                <small class="d-block text-muted">将优先使用此提取结果</small>
            `;
        }
    }

    // 格式化文件大小
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // 设置表单提交
    setupForms() {
        // 论文与任务比较
        document.getElementById('paperTaskForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.submitPaperTaskAnalysis();
        });

        // 论文与论文比较
        document.getElementById('paperPaperForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.submitPaperPaperAnalysis();
        });

        // 关系覆盖率分析
        document.getElementById('relationCoverageForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.submitRelationCoverageAnalysis();
        });
    }

    // 提交论文与任务比较分析
    async submitPaperTaskAnalysis() {
        const taskId = document.getElementById('taskSelect').value;
        const paperFile = document.getElementById('paperFile1').files[0];
        const extractedFile = document.getElementById('extractedFile1').files[0];
        const model = document.getElementById('modelSelect1').value;

        if (!taskId) {
            this.showError('请选择任务');
            return;
        }

        if (!paperFile && !extractedFile) {
            this.showError('请上传论文PDF文件或提取结果JSON文件');
            return;
        }

        const formData = new FormData();
        formData.append('task_id', taskId);
        formData.append('model', model);

        if (extractedFile) {
            formData.append('extracted_file', extractedFile);
            console.log('使用提取结果文件:', extractedFile.name);
        }

        if (paperFile) {
            formData.append('paper_file', paperFile);
        }

        this.startAnalysis(1, '论文与任务比较分析');

        try {
            const response = await fetch('/api/analysis/paper-task', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (data.success) {
                this.monitorAnalysis(data.analysis_id, 1);
            } else {
                this.showAnalysisError(1, data.message);
            }
        } catch (error) {
            this.showAnalysisError(1, '网络错误: ' + error.message);
        }
    }

    // 提交论文与论文比较分析
    async submitPaperPaperAnalysis() {
        const paper1File = document.getElementById('paperFile2').files[0];
        const paper2File = document.getElementById('paperFile3').files[0];
        const extractedFile = document.getElementById('extractedFile2').files[0];
        const model = document.getElementById('modelSelect2').value;

        if (!paper2File) {
            this.showError('请上传论文2文件');
            return;
        }

        if (!paper1File && !extractedFile) {
            this.showError('请上传论文1文件或提取结果JSON文件');
            return;
        }

        const formData = new FormData();
        formData.append('paper2_file', paper2File);
        formData.append('model', model);

        if (extractedFile) {
            formData.append('extracted_file', extractedFile);
            console.log('使用论文1提取结果文件:', extractedFile.name);
        }

        if (paper1File) {
            formData.append('paper1_file', paper1File);
        }

        this.startAnalysis(2, '论文与论文比较分析');

        try {
            const response = await fetch('/api/analysis/paper-paper', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (data.success) {
                this.monitorAnalysis(data.analysis_id, 2);
            } else {
                this.showAnalysisError(2, data.message);
            }
        } catch (error) {
            this.showAnalysisError(2, '网络错误: ' + error.message);
        }
    }

    // 提交关系覆盖率分析
    async submitRelationCoverageAnalysis() {
        const taskId = document.getElementById('taskSelect2').value;
        const model = document.getElementById('modelSelect3').value;

        if (!taskId) {
            this.showError('请选择任务');
            return;
        }

        this.startAnalysis(3, '关系覆盖率分析');

        try {
            const response = await fetch('/api/analysis/task-relation-coverage', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    task_id: taskId,
                    model: model
                })
            });

            const data = await response.json();
            
            if (data.success) {
                this.monitorAnalysis(data.analysis_id, 3);
            } else {
                this.showAnalysisError(3, data.message);
            }
        } catch (error) {
            this.showAnalysisError(3, '网络错误: ' + error.message);
        }
    }

    // 开始分析
    startAnalysis(analysisType, analysisName) {
        const progressContainer = document.getElementById(`progress${analysisType}`);
        const resultContainer = document.getElementById(`result${analysisType}`);
        
        progressContainer.style.display = 'block';
        resultContainer.style.display = 'none';
        
        this.updateProgress(analysisType, 0, `正在启动${analysisName}...`);
    }

    // 监控分析进度
    async monitorAnalysis(analysisId, analysisType) {
        const maxAttempts = 120; // 最多监控10分钟
        let attempts = 0;

        const checkProgress = async () => {
            try {
                const response = await fetch(`/api/analysis/${analysisId}/results`);
                const data = await response.json();

                if (data.success) {
                    this.updateProgress(analysisType, data.progress, data.current_stage);

                    if (data.status === 'completed') {
                        this.showAnalysisResult(analysisType, data.results);
                        return;
                    } else if (data.status === 'failed') {
                        this.showAnalysisError(analysisType, data.message);
                        return;
                    }
                }

                attempts++;
                if (attempts < maxAttempts) {
                    setTimeout(checkProgress, 5000); // 每5秒检查一次
                } else {
                    this.showAnalysisError(analysisType, '分析超时');
                }
            } catch (error) {
                this.showAnalysisError(analysisType, '监控分析进度时出错: ' + error.message);
            }
        };

        checkProgress();
    }

    // 更新进度
    updateProgress(analysisType, progress, message) {
        document.getElementById(`progressText${analysisType}`).textContent = message;
        document.getElementById(`progressPercent${analysisType}`).textContent = `${progress}%`;
        document.getElementById(`progressBar${analysisType}`).style.width = `${progress}%`;
    }

    // 显示分析结果
    showAnalysisResult(analysisType, results) {
        const progressContainer = document.getElementById(`progress${analysisType}`);
        const resultContainer = document.getElementById(`result${analysisType}`);
        const resultContent = document.getElementById(`resultContent${analysisType}`);

        progressContainer.style.display = 'none';
        resultContainer.style.display = 'block';

        // 格式化结果显示
        let html = '<div class="alert alert-success">分析完成！</div>';
        
        if (results.coverage_analysis) {
            const coverage = results.coverage_analysis;
            html += this.formatCoverageResults(coverage);
        } else if (results.relation_coverage) {
            html += this.formatRelationResults(results.relation_coverage);
        }

        resultContent.innerHTML = html;
    }

    // 格式化覆盖率结果
    formatCoverageResults(coverage) {
        let html = '<div class="row">';
        
        if (coverage.method_coverage) {
            const mc = coverage.method_coverage;
            html += `
                <div class="col-md-6">
                    <h6>方法覆盖率</h6>
                    <div class="progress mb-2">
                        <div class="progress-bar bg-info" style="width: ${(mc.coverage_ratio * 100).toFixed(1)}%">
                            ${(mc.coverage_ratio * 100).toFixed(1)}%
                        </div>
                    </div>
                    <small>匹配方法: ${mc.matched_methods}/${mc.total_reference_methods}</small>
                </div>
            `;
        }

        if (coverage.algorithm_coverage) {
            const ac = coverage.algorithm_coverage;
            html += `
                <div class="col-md-6">
                    <h6>算法覆盖率</h6>
                    <div class="progress mb-2">
                        <div class="progress-bar bg-success" style="width: ${(ac.average_coverage_ratio * 100).toFixed(1)}%">
                            ${(ac.average_coverage_ratio * 100).toFixed(1)}%
                        </div>
                    </div>
                    <small>匹配算法: ${ac.matched_algorithms}/${ac.total_reference_algorithms}</small>
                </div>
            `;
        }

        html += '</div>';

        if (coverage.detailed_analysis) {
            html += `
                <div class="mt-3">
                    <h6>详细分析</h6>
                    <div class="alert alert-light">
                        ${coverage.detailed_analysis}
                    </div>
                </div>
            `;
        }

        return html;
    }

    // 格式化关系结果
    formatRelationResults(relationCoverage) {
        const rc = relationCoverage.relation_coverage;
        let html = `
            <div class="row">
                <div class="col-12">
                    <h6>关系覆盖率</h6>
                    <div class="progress mb-3">
                        <div class="progress-bar bg-warning" style="width: ${(rc.overall_coverage_ratio * 100).toFixed(1)}%">
                            ${(rc.overall_coverage_ratio * 100).toFixed(1)}%
                        </div>
                    </div>
                    <p><small>重合关系: ${rc.overlapping_relations}/${rc.total_review_relations}</small></p>
                </div>
            </div>
        `;

        if (rc.coverage_by_type) {
            html += '<div class="row">';
            Object.entries(rc.coverage_by_type).forEach(([type, ratio]) => {
                html += `
                    <div class="col-6 mb-2">
                        <small>${type}: ${(ratio * 100).toFixed(1)}%</small>
                        <div class="progress" style="height: 8px;">
                            <div class="progress-bar bg-secondary" style="width: ${(ratio * 100).toFixed(1)}%"></div>
                        </div>
                    </div>
                `;
            });
            html += '</div>';
        }

        if (relationCoverage.detailed_analysis) {
            html += `
                <div class="mt-3">
                    <h6>详细分析</h6>
                    <div class="alert alert-light">
                        ${relationCoverage.detailed_analysis}
                    </div>
                </div>
            `;
        }

        return html;
    }

    // 显示分析错误
    showAnalysisError(analysisType, message) {
        const progressContainer = document.getElementById(`progress${analysisType}`);
        const resultContainer = document.getElementById(`result${analysisType}`);
        const resultContent = document.getElementById(`resultContent${analysisType}`);

        progressContainer.style.display = 'none';
        resultContainer.style.display = 'block';

        resultContent.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle"></i> 分析失败: ${message}
            </div>
        `;
    }

    // 显示错误消息
    showError(message) {
        this.showAlert(message, 'danger');
    }

    // 显示信息消息
    showInfo(message) {
        this.showAlert(message, 'info');
    }

    // 显示成功消息
    showSuccess(message) {
        this.showAlert(message, 'success');
    }

    // 通用的提示消息显示方法
    showAlert(message, type = 'info') {
        // 创建临时提示
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alert.style.top = '20px';
        alert.style.right = '20px';
        alert.style.zIndex = '9999';
        alert.style.maxWidth = '400px';
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(alert);

        // 根据消息类型设置不同的自动消失时间
        const timeout = type === 'danger' ? 5000 : 3000;
        setTimeout(() => {
            if (alert.parentNode) {
                alert.parentNode.removeChild(alert);
            }
        }, timeout);
    }

    // 加载分析历史记录
    async loadAnalysisHistory() {
        try {
            const tbody = document.getElementById('analysisHistoryTableBody');
            tbody.innerHTML = '<tr><td colspan="5" class="text-center"><i class="fas fa-spinner fa-spin"></i> 加载中...</td></tr>';

            const response = await fetch('/api/analysis/history');
            const data = await response.json();

            if (data.success && data.analyses && data.analyses.length > 0) {
                let html = '';

                data.analyses.forEach(analysis => {
                    const statusClass = this.getAnalysisStatusClass(analysis.status);
                    const formattedDate = this.formatDate(analysis.created_time);

                    html += `
                        <tr>
                            <td>
                                <span class="badge bg-info">${analysis.analysis_type}</span>
                            </td>
                            <td>
                                <small class="text-muted">${analysis.task_id.substring(0, 8)}...</small>
                            </td>
                            <td>
                                <span class="badge ${statusClass}">${analysis.status}</span>
                            </td>
                            <td>${formattedDate}</td>
                            <td>
                                <button class="btn btn-sm btn-primary" onclick="paperAnalysisSystem.viewAnalysisResult('${analysis.analysis_id}')">
                                    查看结果
                                </button>
                            </td>
                        </tr>
                    `;
                });

                tbody.innerHTML = html;
            } else {
                tbody.innerHTML = '<tr><td colspan="5" class="text-center">暂无分析记录</td></tr>';
            }
        } catch (error) {
            console.error('加载分析历史记录失败:', error);
            document.getElementById('analysisHistoryTableBody').innerHTML =
                '<tr><td colspan="5" class="text-center text-danger">加载失败，请稍后重试</td></tr>';
        }
    }

    // 设置历史记录刷新按钮
    setupHistoryRefresh() {
        const refreshBtn = document.getElementById('refreshAnalysisHistoryBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.loadAnalysisHistory();
            });
        }
    }

    // 获取分析状态对应的样式类
    getAnalysisStatusClass(status) {
        switch(status) {
            case '已完成':
                return 'bg-success';
            case '进行中':
                return 'bg-info';
            case '失败':
                return 'bg-danger';
            default:
                return 'bg-secondary';
        }
    }

    // 格式化日期
    formatDate(dateString) {
        if (!dateString || dateString === null || dateString === 'null' || dateString === 'N/A') {
            return '未知时间';
        }

        const date = new Date(dateString);
        if (isNaN(date.getTime())) {
            console.warn('无法解析日期:', dateString);
            return '未知时间';
        }

        return date.toLocaleString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    // 查看分析结果
    async viewAnalysisResult(analysisId) {
        try {
            const response = await fetch(`/api/analysis/${analysisId}/results`);
            const data = await response.json();

            if (data.success && data.results) {
                // 创建模态框显示结果
                this.showAnalysisResultModal(analysisId, data.results);
            } else {
                this.showError('无法加载分析结果: ' + (data.message || '未知错误'));
            }
        } catch (error) {
            console.error('查看分析结果失败:', error);
            this.showError('查看分析结果失败: ' + error.message);
        }
    }

    // 显示分析结果模态框
    showAnalysisResultModal(analysisId, results) {
        // 创建模态框HTML
        const modalHtml = `
            <div class="modal fade" id="analysisResultModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">分析结果 - ${analysisId.substring(0, 8)}...</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            ${this.formatAnalysisResults(results)}
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">关闭</button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // 移除已存在的模态框
        const existingModal = document.getElementById('analysisResultModal');
        if (existingModal) {
            existingModal.remove();
        }

        // 添加新模态框
        document.body.insertAdjacentHTML('beforeend', modalHtml);

        // 显示模态框
        const modal = new bootstrap.Modal(document.getElementById('analysisResultModal'));
        modal.show();
    }

    // 格式化分析结果
    formatAnalysisResults(results) {
        let html = '';

        if (results.coverage_analysis) {
            html += this.formatCoverageResults(results.coverage_analysis);
        } else if (results.relation_coverage) {
            html += this.formatRelationResults(results.relation_coverage);
        } else {
            html += '<div class="alert alert-info">分析结果格式未知</div>';
        }

        return html;
    }
}

// 全局变量，供HTML中的onclick使用
let paperAnalysisSystem;

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    paperAnalysisSystem = new PaperAnalysisSystem();
});
