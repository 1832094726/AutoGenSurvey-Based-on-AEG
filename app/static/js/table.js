// 修改获取实体数据的函数
function fetchEntities() {
    showLoading();
    
    // 使用时间戳防止缓存，但只发送一次请求
    const timestamp = new Date().getTime();
    
    fetch(`/api/entities?_=${timestamp}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`网络请求失败: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("API返回原始数据:", data);
            
            // 提取各类实体
            const algorithms = [];
            const datasets = [];
            const metrics = [];
            
            // 处理不同格式的响应数据
            if (data && data.entities && Array.isArray(data.entities)) {
                // 标准格式: {success: true, entities: [...]}
                data.entities.forEach(entity => {
                    if (entity.entity_type === 'Algorithm') {
                        algorithms.push(entity);
                    } else if (entity.entity_type === 'Dataset') {
                        datasets.push(entity);
                    } else if (entity.entity_type === 'Metric') {
                        metrics.push(entity);
                    } else if (entity.algorithm_entity) {
                        algorithms.push(entity.algorithm_entity);
                    } else if (entity.dataset_entity) {
                        datasets.push(entity.dataset_entity);
                    } else if (entity.metric_entity) {
                        metrics.push(entity.metric_entity);
                    }
                });
            } else if (Array.isArray(data)) {
                // 直接返回数组格式
                data.forEach(entity => {
                    if (entity.entity_type === 'Algorithm') {
                        algorithms.push(entity);
                    } else if (entity.entity_type === 'Dataset') {
                        datasets.push(entity);
                    } else if (entity.entity_type === 'Metric') {
                        metrics.push(entity);
                    } else if (entity.algorithm_entity) {
                        algorithms.push(entity.algorithm_entity);
                    } else if (entity.dataset_entity) {
                        datasets.push(entity.dataset_entity);
                    } else if (entity.metric_entity) {
                        metrics.push(entity.metric_entity);
                    }
                });
            }
            
            console.log("获取到算法数据:", algorithms);
            console.log("获取到数据集数据:", datasets);
            console.log("获取到评估指标数据:", metrics);
            
            // 更新表格显示
            updateAlgorithmTable(algorithms);
            updateDatasetTable(datasets);
            updateMetricTable(metrics);
            
            hideLoading();
        })
        .catch(error => {
            console.error("获取实体数据失败:", error);
            showError("获取实体数据失败，请检查网络连接和服务器状态");
            hideLoading();
        });
}

// 更新算法表格
function updateAlgorithmTable(algorithms) {
    const tableContainer = document.getElementById('algorithm-table-container');
    if (!tableContainer) {
        console.error("找不到算法表格容器");
        return;
    }
    
    if (!algorithms || algorithms.length === 0) {
        tableContainer.innerHTML = '<div class="alert alert-info">暂无算法数据</div>';
        return;
    }
    
    // 创建表格HTML
    let html = `
        <table class="table table-striped table-bordered">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>名称</th>
                    <th>年份</th>
                    <th>任务</th>
                    <th>作者</th>
                    <th>操作</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    // 添加行数据
    algorithms.forEach(algo => {
        html += `
            <tr>
                <td>${algo.algorithm_id || algo.entity_id || ''}</td>
                <td>${algo.name || ''}</td>
                <td>${algo.year || ''}</td>
                <td>${algo.task || ''}</td>
                <td>${formatAuthors(algo.authors)}</td>
                <td>
                    <button class="btn btn-sm btn-info" onclick="viewEntity('${algo.algorithm_id || algo.entity_id}')">查看</button>
                </td>
            </tr>
        `;
    });
    
    html += `
            </tbody>
        </table>
    `;
    
    tableContainer.innerHTML = html;
}

// 更新数据集表格
function updateDatasetTable(datasets) {
    const tableContainer = document.getElementById('dataset-table-container');
    if (!tableContainer) {
        console.error("找不到数据集表格容器");
        return;
    }
    
    if (!datasets || datasets.length === 0) {
        tableContainer.innerHTML = '<div class="alert alert-info">暂无数据集数据</div>';
        return;
    }
    
    // 创建表格HTML
    let html = `
        <table class="table table-striped table-bordered">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>名称</th>
                    <th>描述</th>
                    <th>领域</th>
                    <th>年份</th>
                    <th>操作</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    // 添加行数据
    datasets.forEach(dataset => {
        html += `
            <tr>
                <td>${dataset.dataset_id || dataset.entity_id || ''}</td>
                <td>${dataset.name || ''}</td>
                <td>${truncateText(dataset.description || '', 100)}</td>
                <td>${dataset.domain || ''}</td>
                <td>${dataset.year || ''}</td>
                <td>
                    <button class="btn btn-sm btn-info" onclick="viewEntity('${dataset.dataset_id || dataset.entity_id}')">查看</button>
                </td>
            </tr>
        `;
    });
    
    html += `
            </tbody>
        </table>
    `;
    
    tableContainer.innerHTML = html;
}

// 更新评估指标表格
function updateMetricTable(metrics) {
    const tableContainer = document.getElementById('metric-table-container');
    if (!tableContainer) {
        console.error("找不到评估指标表格容器");
        return;
    }
    
    if (!metrics || metrics.length === 0) {
        tableContainer.innerHTML = '<div class="alert alert-info">暂无评估指标数据</div>';
        return;
    }
    
    // 创建表格HTML
    let html = `
        <table class="table table-striped table-bordered">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>名称</th>
                    <th>描述</th>
                    <th>分类</th>
                    <th>公式</th>
                    <th>操作</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    // 添加行数据
    metrics.forEach(metric => {
        html += `
            <tr>
                <td>${metric.metric_id || metric.entity_id || ''}</td>
                <td>${metric.name || ''}</td>
                <td>${truncateText(metric.description || '', 100)}</td>
                <td>${metric.category || ''}</td>
                <td>${metric.formula || ''}</td>
                <td>
                    <button class="btn btn-sm btn-info" onclick="viewEntity('${metric.metric_id || metric.entity_id}')">查看</button>
                </td>
            </tr>
        `;
    });
    
    html += `
            </tbody>
        </table>
    `;
    
    tableContainer.innerHTML = html;
}

// 格式化作者数组
function formatAuthors(authors) {
    if (!authors) return '';
    if (typeof authors === 'string') return authors;
    if (!Array.isArray(authors)) return '';
    if (authors.length === 0) return '';
    
    if (authors.length <= 3) {
        return authors.join(', ');
    } else {
        return authors.slice(0, 3).join(', ') + ' 等';
    }
}

// 截断长文本
function truncateText(text, maxLength) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// 查看实体详情
function viewEntity(entityId) {
    window.location.href = `/entities/${entityId}`;
}

// 添加一个简单的加载和错误提示功能
function showLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.id = 'loading-indicator';
    loadingDiv.innerHTML = '加载中...';
    loadingDiv.style.position = 'fixed';
    loadingDiv.style.top = '50%';
    loadingDiv.style.left = '50%';
    loadingDiv.style.transform = 'translate(-50%, -50%)';
    loadingDiv.style.padding = '10px 20px';
    loadingDiv.style.background = 'rgba(0,0,0,0.7)';
    loadingDiv.style.color = 'white';
    loadingDiv.style.borderRadius = '5px';
    loadingDiv.style.zIndex = '9999';
    document.body.appendChild(loadingDiv);
}

function hideLoading() {
    const loadingDiv = document.getElementById('loading-indicator');
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.id = 'error-message';
    errorDiv.innerHTML = message;
    errorDiv.style.position = 'fixed';
    errorDiv.style.top = '10px';
    errorDiv.style.left = '50%';
    errorDiv.style.transform = 'translateX(-50%)';
    errorDiv.style.padding = '10px 20px';
    errorDiv.style.background = 'rgba(255,0,0,0.8)';
    errorDiv.style.color = 'white';
    errorDiv.style.borderRadius = '5px';
    errorDiv.style.zIndex = '9999';
    document.body.appendChild(errorDiv);
    
    setTimeout(() => {
        const div = document.getElementById('error-message');
        if (div) div.remove();
    }, 5000);
}

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    fetchEntities();
}); 