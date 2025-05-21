/**
 * 调用API生成算法演化关系
 */
function generateRelations() {
    // 显示加载状态
    showLoading('正在生成算法演化关系...');
    
    // 调用API
    fetch('/api/v1/generate_relations', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            max_pairs: 200,  // 默认处理200个实体对
            batch_size: 5    // 每批5个
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 成功启动任务，显示任务ID
            showNotification('已开始生成算法演化关系，任务ID: ' + data.task_id, 'success');
            
            // 开始轮询任务状态
            pollTaskStatus(data.task_id);
        } else {
            // 显示错误消息
            showNotification('生成关系失败: ' + data.message, 'error');
            hideLoading();
        }
    })
    .catch(error => {
        console.error('调用生成关系API出错:', error);
        showNotification('系统错误，请稍后再试', 'error');
        hideLoading();
    });
}

/**
 * 轮询任务状态
 * @param {string} taskId 任务ID
 */
function pollTaskStatus(taskId) {
    // 设置轮询间隔（秒）
    const pollInterval = 5000;
    
    // 获取任务状态
    fetch('/api/v1/task_status/' + taskId)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'completed') {
                // 任务完成
                hideLoading();
                showNotification('算法演化关系生成完成!', 'success');
                
                // 重新加载关系数据
                setTimeout(() => {
                    loadGraphData();
                }, 1000);
            } else if (data.status === 'failed') {
                // 任务失败
                hideLoading();
                showNotification('生成关系失败: ' + data.message, 'error');
            } else {
                // 任务仍在进行中，继续轮询
                updateLoadingMessage('正在生成关系: ' + data.current_stage + ' (' + Math.round(data.progress * 100) + '%)');
                setTimeout(() => pollTaskStatus(taskId), pollInterval);
            }
        })
        .catch(error => {
            console.error('获取任务状态出错:', error);
            hideLoading();
            showNotification('获取任务状态失败，请检查网络连接', 'error');
        });
}

/**
 * 显示通知消息
 * @param {string} message 消息内容
 * @param {string} type 消息类型：success, info, warning, error
 */
function showNotification(message, type = 'info') {
    // 检查是否有通知容器，如果没有则创建
    let container = document.getElementById('notification-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'notification-container';
        container.style.position = 'fixed';
        container.style.top = '20px';
        container.style.right = '20px';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
    }
    
    // 创建通知元素
    const notification = document.createElement('div');
    notification.className = 'notification ' + type;
    notification.style.padding = '12px 20px';
    notification.style.marginBottom = '10px';
    notification.style.borderRadius = '4px';
    notification.style.boxShadow = '0 2px 10px rgba(0,0,0,0.1)';
    notification.style.minWidth = '300px';
    notification.style.animation = 'fadeIn 0.3s';
    
    // 设置背景颜色
    switch (type) {
        case 'success':
            notification.style.backgroundColor = '#4CAF50';
            notification.style.color = 'white';
            break;
        case 'error':
            notification.style.backgroundColor = '#F44336';
            notification.style.color = 'white';
            break;
        case 'warning':
            notification.style.backgroundColor = '#FF9800';
            notification.style.color = 'white';
            break;
        default:
            notification.style.backgroundColor = '#2196F3';
            notification.style.color = 'white';
    }
    
    // 设置消息内容
    notification.textContent = message;
    
    // 添加到容器
    container.appendChild(notification);
    
    // 3秒后自动移除
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transition = 'opacity 0.5s';
        setTimeout(() => {
            if (notification.parentNode === container) {
                container.removeChild(notification);
            }
        }, 500);
    }, 3000);
}

/**
 * 显示加载状态
 * @param {string} message 加载提示信息
 */
function showLoading(message) {
    // 检查是否已有加载层
    let loadingOverlay = document.getElementById('loading-overlay');
    if (!loadingOverlay) {
        // 创建半透明背景层
        loadingOverlay = document.createElement('div');
        loadingOverlay.id = 'loading-overlay';
        loadingOverlay.style.position = 'fixed';
        loadingOverlay.style.top = '0';
        loadingOverlay.style.left = '0';
        loadingOverlay.style.width = '100%';
        loadingOverlay.style.height = '100%';
        loadingOverlay.style.backgroundColor = 'rgba(0,0,0,0.5)';
        loadingOverlay.style.display = 'flex';
        loadingOverlay.style.justifyContent = 'center';
        loadingOverlay.style.alignItems = 'center';
        loadingOverlay.style.zIndex = '9998';
        
        // 创建加载内容区
        const loadingContent = document.createElement('div');
        loadingContent.style.backgroundColor = 'white';
        loadingContent.style.padding = '20px';
        loadingContent.style.borderRadius = '8px';
        loadingContent.style.boxShadow = '0 4px 20px rgba(0,0,0,0.2)';
        loadingContent.style.textAlign = 'center';
        
        // 创建加载图标
        const spinner = document.createElement('div');
        spinner.className = 'loading-spinner';
        spinner.style.borderRadius = '50%';
        spinner.style.width = '40px';
        spinner.style.height = '40px';
        spinner.style.margin = '0 auto 15px';
        spinner.style.border = '4px solid #f3f3f3';
        spinner.style.borderTop = '4px solid #3498db';
        spinner.style.animation = 'spin 1s linear infinite';
        
        // 创建消息元素
        const msgElement = document.createElement('div');
        msgElement.id = 'loading-message';
        msgElement.textContent = message;
        
        // 添加元素
        loadingContent.appendChild(spinner);
        loadingContent.appendChild(msgElement);
        loadingOverlay.appendChild(loadingContent);
        
        // 添加动画样式
        const style = document.createElement('style');
        style.textContent = '@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }';
        document.head.appendChild(style);
        
        // 添加到页面
        document.body.appendChild(loadingOverlay);
    } else {
        // 更新消息
        const msgElement = document.getElementById('loading-message');
        if (msgElement) {
            msgElement.textContent = message;
        }
    }
}

/**
 * 更新加载状态消息
 * @param {string} message 新的加载提示信息
 */
function updateLoadingMessage(message) {
    const msgElement = document.getElementById('loading-message');
    if (msgElement) {
        msgElement.textContent = message;
    }
}

/**
 * 隐藏加载状态
 */
function hideLoading() {
    const loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay) {
        document.body.removeChild(loadingOverlay);
    }
}

// 添加"生成关系"按钮的事件监听器
document.addEventListener('DOMContentLoaded', function() {
    // 寻找或创建生成关系按钮
    let generateBtn = document.getElementById('generate-relations-btn');
    if (!generateBtn) {
        // 创建按钮
        generateBtn = document.createElement('button');
        generateBtn.id = 'generate-relations-btn';
        generateBtn.textContent = '自动生成演化关系';
        generateBtn.className = 'btn btn-primary';
        generateBtn.style.position = 'fixed';
        generateBtn.style.top = '10px';
        generateBtn.style.right = '10px';
        generateBtn.style.zIndex = '1000';
        
        // 添加到工具栏或页面
        const toolbar = document.querySelector('.toolbar') || document.body;
        toolbar.appendChild(generateBtn);
    }
    
    // 添加点击事件
    generateBtn.addEventListener('click', generateRelations);
}); 