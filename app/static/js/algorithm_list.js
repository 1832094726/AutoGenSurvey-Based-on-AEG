// 加载算法实体列表
function loadAlgorithmList() {
    $('#algorithm-list-container').html('<div class="text-center my-5"><div class="spinner-border" role="status"></div><p class="mt-2">正在加载算法实体...</p></div>');
    
    // 添加调试信息
    console.log('开始加载算法实体列表...');
    
    $.ajax({
        url: '/api/entities',  // 修改为正确的API路径
        method: 'GET',
        timeout: 30000,  // 30秒超时
        success: function(response) {
            console.log('成功获取算法实体数据:', response);
            
            // 尝试从不同的响应格式中提取实体数据
            var entities = [];
            
            if (response.success === true) {
                // 标准格式: {success: true, entities: [...]}
                if (response.entities && Array.isArray(response.entities)) {
                    entities = response.entities;
                }
                // 可能的替代格式: {success: true, algorithms: [...]}
                else if (response.algorithms && Array.isArray(response.algorithms)) {
                    entities = response.algorithms;
                }
            } 
            // 直接返回数组格式
            else if (Array.isArray(response)) {
                entities = response;
            }
            // 可能的其他格式
            else if (typeof response === 'object') {
                // 尝试从response中找到第一个数组属性
                for (var key in response) {
                    if (Array.isArray(response[key])) {
                        entities = response[key];
                        break;
                    }
                }
            }
            
            console.log(`提取到 ${entities.length} 个实体`);
            
            if (entities.length > 0) {
                // 创建表格
                var tableHtml = '<table class="table table-striped table-bordered data-table" id="algorithm-table">';
                tableHtml += '<thead><tr>';
                tableHtml += '<th>ID</th>';
                tableHtml += '<th>名称</th>';
                tableHtml += '<th>年份</th>';
                tableHtml += '<th>任务</th>';
                tableHtml += '<th>作者</th>';
                tableHtml += '<th>数据集</th>';
                tableHtml += '<th>评价指标</th>';
                tableHtml += '<th>操作</th>';
                tableHtml += '</tr></thead>';
                tableHtml += '<tbody>';
                
                // 添加行
                var rowCount = 0;
                entities.forEach(function(item) {
                    // 处理不同的数据结构
                    var entity = null;
                    
                    if (item.algorithm_entity) {
                        entity = item.algorithm_entity;
                    } else if (item.id || item.algorithm_id || item.name) {
                        // 可能实体直接是顶层对象
                        entity = item;
                    }
                    
                    if (entity) {
                        rowCount++;
                        var id = entity.algorithm_id || entity.id || '';
                        
                        tableHtml += '<tr>';
                        tableHtml += '<td>' + id + '</td>';
                        tableHtml += '<td>' + (entity.name || '') + '</td>';
                        tableHtml += '<td>' + (entity.year || '') + '</td>';
                        tableHtml += '<td>' + (entity.task || '') + '</td>';
                        tableHtml += '<td>' + formatAuthors(entity.authors) + '</td>';
                        tableHtml += '<td>' + formatArray(entity.dataset) + '</td>';
                        tableHtml += '<td>' + formatArray(entity.metrics) + '</td>';
                        tableHtml += '<td><a href="/entity/' + id + '" class="btn btn-sm btn-info">查看</a></td>';
                        tableHtml += '</tr>';
                    }
                });
                
                tableHtml += '</tbody></table>';
                
                if (rowCount > 0) {
                    // 显示表格
                    $('#algorithm-list-container').html(tableHtml);
                    
                    console.log('生成表格HTML完成，rowCount =', rowCount);
                    
                    // 检查并销毁已存在的DataTable实例
                    if ($.fn.DataTable.isDataTable('#algorithm-table')) {
                        $('#algorithm-table').DataTable().destroy();
                    }
                    
                    // 使用内联方式初始化，避免加载远程语言文件
                    try {
                        console.log('正在初始化DataTable...');
                        // 初始化DataTables之前先验证表格HTML结构
                        if ($('#algorithm-table thead tr th').length === 0) {
                            console.error('表格头部未正确渲染');
                            $('#algorithm-list-container').html('<div class="alert alert-danger">表格结构错误，无法初始化DataTable</div>');
                            return;
                        }
                        
                        $('#algorithm-table').DataTable({
                            // 禁用自动加载远程语言文件
                            language: {
                                "emptyTable": "表中数据为空",
                                "info": "显示第 _START_ 至 _END_ 条记录，共 _TOTAL_ 条",
                                "infoEmpty": "显示第 0 至 0 条记录，共 0 条",
                                "infoFiltered": "(由 _MAX_ 条记录过滤)",
                                "lengthMenu": "显示 _MENU_ 条记录",
                                "loadingRecords": "加载中...",
                                "processing": "处理中...",
                                "search": "搜索:",
                                "zeroRecords": "没有找到匹配的记录",
                                "paginate": {
                                    "first": "首页",
                                    "last": "末页",
                                    "next": "下页",
                                    "previous": "上页"
                                }
                            },
                            responsive: true,
                            paging: true,
                            searching: true,
                            ordering: true
                        });
                        console.log('DataTable初始化成功');
                    } catch (e) {
                        console.error('DataTable初始化失败:', e);
                        $('#algorithm-list-container').append('<div class="alert alert-danger mt-3">表格初始化出错: ' + e.message + '</div>');
                    }
                } else {
                    $('#algorithm-list-container').html('<div class="alert alert-warning">找到了实体数据，但格式不符合预期。</div>');
                    console.warn('找到了实体数据，但格式不符合预期:', entities);
                }
            } else {
                // 无数据或加载失败
                $('#algorithm-list-container').html('<div class="alert alert-info">暂无算法实体数据。请上传PDF文件进行处理。</div>');
            }
        },
        error: function(xhr, status, error) {
            console.error('加载算法实体失败:', error);
            console.error('状态码:', xhr.status);
            console.error('响应文本:', xhr.responseText);
            
            // 显示错误信息
            $('#algorithm-list-container').html('<div class="alert alert-danger">加载数据失败: ' + error + '<br>请检查服务器日志或刷新页面重试。</div>');
        }
    });
}

// 格式化数组为逗号分隔的字符串
function formatArray(arr) {
    if (!arr || !Array.isArray(arr) || arr.length === 0) return '';
    return arr.join(', ');
}

// 格式化作者数组
function formatAuthors(authors) {
    if (!authors || !Array.isArray(authors) || authors.length === 0) return '';
    if (authors.length <= 3) return authors.join(', ');
    return authors.slice(0, 3).join(', ') + ' 等';
}

// 页面加载时执行
$(document).ready(function() {
    loadAlgorithmList();
}); 