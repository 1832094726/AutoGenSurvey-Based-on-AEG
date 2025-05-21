// 文件上传和处理相关逻辑
document.addEventListener('DOMContentLoaded', function() {
    // 单文件上传处理
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const uploadProgress = document.getElementById('uploadProgress');
    const progressBar = document.getElementById('progressBar');
    const uploadStatus = document.getElementById('uploadStatus');
    const processingStage = document.getElementById('processingStage');
    const currentFile = document.getElementById('currentFile');
    
    // 批量上传处理
    const batchUploadForm = document.getElementById('batchUploadForm');
    const batchFileInput = document.getElementById('batchFileInput');
    const batchProgress = document.getElementById('batchProgress');
    const batchProgressBar = document.getElementById('batchProgressBar');
    const batchStatus = document.getElementById('batchStatus');
    const batchProcessingStage = document.getElementById('batchProcessingStage');
    const batchCurrentFile = document.getElementById('batchCurrentFile');
    
    // 全局任务状态变量
    let currentTaskId = null;
    let statusCheckInterval = null;
    let errorCount = 0;
    
    // 单文件上传处理
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            if (!fileInput.files.length) {
                alert('请选择要上传的文件');
                return;
            }
            
            // 显示进度条
            uploadProgress.style.display = 'block';
            uploadStatus.textContent = '正在上传文件...';
            processingStage.textContent = '准备处理';
            currentFile.textContent = fileInput.files[0].name;
            progressBar.style.width = '10%';
            
            // 创建表单数据
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            // 发送请求
            fetch('/api/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    uploadStatus.textContent = '文件上传成功，正在处理...';
                    
                    // 保存任务ID并启动状态检查
                    currentTaskId = data.task_id;
                    startStatusCheck(currentTaskId);
                } else {
                    uploadStatus.textContent = `上传失败: ${data.message}`;
                    progressBar.style.width = '0%';
                    
                    // 如果有任务ID，仍然启动状态检查
                    if (data.task_id) {
                        currentTaskId = data.task_id;
                        startStatusCheck(currentTaskId);
                    }
                }
            })
            .catch(error => {
                console.error('上传出错:', error);
                uploadStatus.textContent = `上传出错: ${error.message}`;
                progressBar.style.width = '0%';
            });
        });
    }
    
    // 批量文件上传处理
    if (batchUploadForm) {
        batchUploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            if (!batchFileInput.files.length) {
                alert('请选择要批量上传的文件');
                return;
            }
            
            // 显示进度条
            batchProgress.style.display = 'block';
            batchStatus.textContent = '正在上传文件...';
            batchProcessingStage.textContent = '准备处理';
            batchCurrentFile.textContent = `选择了 ${batchFileInput.files.length} 个文件`;
            batchProgressBar.style.width = '10%';
            
            // 创建表单数据
            const formData = new FormData();
            
            // 添加所有文件
            for (let i = 0; i < batchFileInput.files.length; i++) {
                formData.append('files[]', batchFileInput.files[i]);
            }
            
            // 发送请求
            fetch('/api/batch_upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    batchStatus.textContent = '文件上传成功，正在批量处理...';
                    
                    // 保存任务ID并启动状态检查
                    currentTaskId = data.task_id;
                    startStatusCheck(currentTaskId, true);
                } else {
                    batchStatus.textContent = `批量上传失败: ${data.message}`;
                    batchProgressBar.style.width = '0%';
                    
                    // 如果有任务ID，仍然启动状态检查
                    if (data.task_id) {
                        currentTaskId = data.task_id;
                        startStatusCheck(currentTaskId, true);
                    }
                }
            })
            .catch(error => {
                console.error('批量上传出错:', error);
                batchStatus.textContent = `批量上传出错: ${error.message}`;
                batchProgressBar.style.width = '0%';
            });
        });
    }
    
    // 定期检查处理状态
    function startStatusCheck(taskId, isBatch = false) {
        // 清除现有的状态检查
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
        }
        
        // 重置错误计数
        errorCount = 0;
        
        // 立即执行一次状态检查
        checkStatus();
        
        // 设置定期检查
        statusCheckInterval = setInterval(checkStatus, 2000); // 每2秒检查一次
        
        function checkStatus() {
            console.log(`检查任务状态: ${taskId}, isBatch: ${isBatch}`);
            
            fetch(`/api/status/${taskId}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    console.log("状态检查返回: ", data);
                    
                    if (data.success) {
                        // 重置错误计数
                        errorCount = 0;
                        
                        updateStatusDisplay(data.status, isBatch);
                        
                        // 如果处理完成或失败，停止检查
                        if (data.status.status === 'completed' || data.status.status === 'failed') {
                            console.log(`任务 ${taskId} ${data.status.status}`);
                            clearInterval(statusCheckInterval);
                            
                            // 如果处理完成，刷新知识图谱
                            if (data.status.status === 'completed') {
                                // 延迟一秒后刷新图谱
                                setTimeout(() => {
                                    refreshKnowledgeGraph();
                                }, 1000);
                            }
                        }
                    } else {
                        console.error('获取状态失败:', data.message);
                        
                        // 如果多次失败，停止检查
                        errorCount++;
                        if (errorCount > 5) {
                            clearInterval(statusCheckInterval);
                            
                            const statusElement = isBatch ? batchStatus : uploadStatus;
                            statusElement.textContent = '无法获取处理状态，请检查日志';
                        }
                    }
                })
                .catch(error => {
                    console.error('状态检查出错:', error);
                    
                    // 如果多次失败，停止检查
                    errorCount++;
                    if (errorCount > 5) {
                        clearInterval(statusCheckInterval);
                        
                        const statusElement = isBatch ? batchStatus : uploadStatus;
                        statusElement.textContent = '状态检查出错，请检查网络连接';
                    }
                });
        }
    }
    
    // 更新状态显示
    function updateStatusDisplay(status, isBatch = false) {
        // 选择要更新的元素
        const statusElement = isBatch ? batchStatus : uploadStatus;
        const stageElement = isBatch ? batchProcessingStage : processingStage;
        const fileElement = isBatch ? batchCurrentFile : currentFile;
        const progressElement = isBatch ? batchProgressBar : progressBar;
        
        // 打印状态信息到控制台
        console.log("更新状态显示:", {
            status: status.status,
            message: status.message,
            stage: status.current_stage,
            file: status.current_file,
            progress: status.progress
        });
        
        // 更新状态显示
        statusElement.textContent = status.message || `状态: ${status.status}`;
        stageElement.textContent = status.current_stage || '未知阶段';
        fileElement.textContent = status.current_file || '无文件信息';
        
        // 更新进度条
        if (status.progress !== null && status.progress !== undefined) {
            const progressPercent = Math.round(status.progress * 100);
            progressElement.style.width = `${progressPercent}%`;
            progressElement.textContent = `${progressPercent}%`;
            
            // 根据状态设置进度条颜色
            if (status.status === 'completed') {
                progressElement.className = 'progress-bar bg-success';
            } else if (status.status === 'failed') {
                progressElement.className = 'progress-bar bg-danger';
            } else if (status.status === 'processing') {
                progressElement.className = 'progress-bar bg-info progress-bar-striped progress-bar-animated';
            } else {
                progressElement.className = 'progress-bar bg-warning progress-bar-striped progress-bar-animated';
            }
        }
    }
    
    // 刷新知识图谱
    function refreshKnowledgeGraph() {
        // 检查是否存在刷新图谱的函数
        if (typeof renderKnowledgeGraph === 'function') {
            // 重新加载数据并渲染图谱
            renderKnowledgeGraph();
        } else {
            console.log('知识图谱刷新函数不可用，将重新加载页面');
            
            // 如果在知识图谱页面，刷新页面
            if (window.location.pathname.includes('knowledge_graph')) {
                window.location.reload();
            } else {
                // 否则跳转到知识图谱页面
                window.location.href = '/knowledge_graph';
            }
        }
    }
    
    // 如果是从上传页面跳转过来，并有任务ID参数，启动状态检查
    const urlParams = new URLSearchParams(window.location.search);
    const taskIdParam = urlParams.get('task_id');
    const isBatchParam = urlParams.get('batch') === 'true';
    
    if (taskIdParam) {
        currentTaskId = taskIdParam;
        startStatusCheck(currentTaskId, isBatchParam);
    }
}); 