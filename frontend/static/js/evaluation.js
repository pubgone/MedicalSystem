// frontend/static/js/evaluation.js
// ==================== 配置 ====================
var API_BASE = 'http://localhost:8000';

// ==================== 页面初始化 ====================
function loadEvaluationPage() {
    console.log('📊 加载测评页面...');
    loadQuestions();
    loadEvaluationReports();
    loadEvaluationMetrics();
}

// ==================== 加载测试问题 ====================
function loadQuestions() {
    fetch(API_BASE + '/evaluation/questions')
        .then(function(response) {
            if (!response.ok) {
                throw new Error('HTTP ' + response.status);
            }
            return response.json();
        })
        .then(function(questions) {
            var tbody = document.getElementById('questionsTableBody');
            if (!tbody) return;
            
            if (questions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="empty">暂无测试问题</td></tr>';
                var countEl = document.getElementById('metricQuestionCount');
                if (countEl) countEl.textContent = '0';
                return;
            }
            
            var html = '';
            for (var i = 0; i < questions.length; i++) {
                var q = questions[i];
                var queryText = q.query.length > 50 ? q.query.substring(0, 50) + '...' : q.query;
                var dept = q.department || '-';
                var badge = getDifficultyBadge(q.difficulty);
                
                html += '<tr>';
                html += '<td>' + q.id + '</td>';
                html += '<td>' + queryText + '</td>';
                html += '<td>' + dept + '</td>';
                html += '<td>' + badge + '</td>';
                html += '<td><div class="action-buttons">';
                html += '<button class="action-btn evaluate" onclick="evaluateSingleQuestion(\'' + q.id + '\')">评估</button>';
                html += '</div></td></tr>';
            }
            
            tbody.innerHTML = html;
            var countEl = document.getElementById('metricQuestionCount');
            if (countEl) countEl.textContent = questions.length;
        })
        .catch(function(error) {
            console.error('❌ 加载问题失败:', error);
            var tbody = document.getElementById('questionsTableBody');
            if (tbody) {
                tbody.innerHTML = '<tr><td colspan="5" class="empty">加载失败</td></tr>';
            }
            alert('加载问题失败：' + error.message);
        });
}

// ==================== 加载评估报告 ====================
function loadEvaluationReports() {
    console.log('🔄 加载评估报告...');
    
    fetch(API_BASE + '/evaluation/reports')
        .then(function(response) {
            if (!response.ok) {
                throw new Error('HTTP ' + response.status);
            }
            return response.json();
        })
        .then(function(reports) {
            var tbody = document.getElementById('reportsTableBody');
            if (!tbody) return;
            
            if (reports.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" class="empty">暂无评估报告</td></tr>';
                return;
            }
            
            var html = '';
            for (var i = 0; i < reports.length; i++) {
                var r = reports[i];
                var score = r.avg_score ? (r.avg_score * 100).toFixed(1) + '%' : '-';
                var time = formatDateTime(r.timestamp);
                var statusBadge = getStatusBadge(r.status);
                
                html += '<tr>';
                html += '<td>' + r.evaluation_id + '</td>';
                html += '<td>' + statusBadge + '</td>';
                html += '<td>' + score + '</td>';
                html += '<td>' + (r.result_count || '-') + '</td>';
                html += '<td>' + time + '</td>';
                html += '<td><div class="action-buttons">';
                html += '<button class="action-btn view" onclick="viewEvaluationDetail(\'' + r.evaluation_id + '\')">查看</button>';
                html += '</div></td></tr>';
            }
            
            tbody.innerHTML = html;
            updateEvaluationMetrics(reports);
        })
        .catch(function(error) {
            console.error('❌ 加载报告失败:', error);
            var tbody = document.getElementById('reportsTableBody');
            if (tbody) {
                tbody.innerHTML = '<tr><td colspan="6" class="empty">加载失败</td></tr>';
            }
            alert('加载报告失败：' + error.message);
        });
}

// ==================== 刷新报告按钮 ====================
function refreshReports() {
    console.log('🔄 手动刷新报告...');
    loadEvaluationReports();
}

// ==================== 更新指标 ====================
function updateEvaluationMetrics(reports) {
    if (reports.length > 0) {
        var totalEl = document.getElementById('metricTotalEval');
        if (totalEl) totalEl.textContent = reports.length;
        
        var sum = 0;
        for (var i = 0; i < reports.length; i++) {
            sum += reports[i].avg_score || 0;
        }
        var avgScore = sum / reports.length;
        
        var avgEl = document.getElementById('metricAvgScore');
        if (avgEl) {
            avgEl.textContent = (avgScore * 100).toFixed(1) + '%';
            if (avgScore >= 0.7) {
                avgEl.className = 'value good';
            } else if (avgScore >= 0.5) {
                avgEl.className = 'value warning';
            } else {
                avgEl.className = 'value bad';
            }
        }
        
        var latestEl = document.getElementById('metricLatestEval');
        if (latestEl) {
            latestEl.textContent = formatDateTime(reports[0].timestamp);
        }
    }
}

// ==================== 辅助函数 ====================
function getDifficultyBadge(difficulty) {
    var colors = {
        'easy': '#27ae60',
        'medium': '#f39c12',
        'hard': '#e74c3c'
    };
    var labels = {
        'easy': '简单',
        'medium': '中等',
        'hard': '困难'
    };
    var color = colors[difficulty] || '#95a5a6';
    var label = labels[difficulty] || difficulty;
    return '<span class="status-badge" style="background:' + color + '20;color:' + color + '">' + label + '</span>';
}

function getStatusBadge(status) {
    var classes = {
        'completed': 'status-badge completed',
        'running': 'status-badge running',
        'failed': 'status-badge failed'
    };
    var labels = {
        'completed': '已完成',
        'running': '进行中',
        'failed': '失败'
    };
    var cls = classes[status] || 'status-badge';
    var label = labels[status] || status;
    return '<span class="' + cls + '">' + label + '</span>';
}

function formatDateTime(isoString) {
    if (!isoString) return '-';
    var date = new Date(isoString);
    return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// ==================== 批量评估 ====================
function startBatchEvaluation() {
    if (!confirm('确定开始批量评估？')) return;
    
    var btn = event.target;
    btn.disabled = true;
    btn.innerHTML = '<span class="loading-spinner"></span> 评估中...';
    
    fetch(API_BASE + '/evaluation/evaluate_batch', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            dataset_name: 'medical_benchmark_v1',
            config: {retrieval_mode: 'hybrid_rerank', top_k: 5}
        })
    })
    .then(function(response) {
        return response.json();
    })
    .then(function(result) {
        if (result.success) {
            alert('✅ 任务已启动！ID: ' + (result.data?.evaluation_id || '未知'));
            setTimeout(loadEvaluationReports, 5000);
        } else {
            alert('❌ 失败：' + result.error);
        }
    })
    .catch(function(error) {
        alert('❌ 请求失败：' + error.message);
    })
    .finally(function() {
        btn.disabled = false;
        btn.innerHTML = '🚀 开始批量评估';
    });
}

// ==================== 单问题评估 ====================
function evaluateSingleQuestion(questionId) {
    if (!confirm('确定评估问题 ' + questionId + '？')) return;
    
    fetch(API_BASE + '/evaluation/evaluate_single?question_id=' + questionId, {
        method: 'POST'
    })
    .then(function(response) {
        return response.json();
    })
    .then(function(result) {
        if (result.success) {
            var score = result.data?.overall_score || 0;
            alert('✅ 得分：' + (score * 100).toFixed(1) + '%');
            loadEvaluationReports();
        } else {
            alert('❌ 失败：' + result.error);
        }
    })
    .catch(function(error) {
        alert('❌ 请求失败：' + error.message);
    });
}

// ==================== 添加问题 ====================
function showAddQuestionModal() {
    var modal = document.getElementById('addQuestionModal');
    if (modal) modal.classList.add('active');
}

function closeAddQuestionModal() {
    var modal = document.getElementById('addQuestionModal');
    if (modal) modal.classList.remove('active');
    var form = document.getElementById('addQuestionForm');
    if (form) form.reset();
}

function submitNewQuestion() {
    var question = {
        id: document.getElementById('questionId').value,
        query: document.getElementById('questionContent').value,
        department: document.getElementById('questionDepartment').value,
        difficulty: document.getElementById('questionDifficulty').value,
        category: document.getElementById('questionCategory').value || 'general'
    };
    
    if (!question.id || !question.query) {
        alert('请填写问题 ID 和内容');
        return;
    }
    
    fetch(API_BASE + '/evaluation/questions', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(question)
    })
    .then(function(response) {
        return response.json();
    })
    .then(function(result) {
        if (result.success) {
            alert('✅ 已添加');
            closeAddQuestionModal();
            loadQuestions();
        } else {
            alert('❌ 失败：' + result.error);
        }
    })
    .catch(function(error) {
        alert('❌ 请求失败：' + error.message);
    });
}

// ==================== 查看详情 ====================
function viewEvaluationDetail(evaluationId) {
    fetch(API_BASE + '/evaluation/report/' + evaluationId)
    .then(function(response) {
        return response.json();
    })
    .then(function(report) {
        var content = document.getElementById('evaluationDetailContent');
        if (content) {
            content.innerHTML = '<p><strong>ID:</strong> ' + report.evaluation_id + '</p>' +
                '<p><strong>状态:</strong> ' + report.status + '</p>' +
                '<p><strong>得分:</strong> ' + ((report.avg_score || 0) * 100).toFixed(1) + '%</p>' +
                '<p><strong>题目数:</strong> ' + (report.results?.length || 0) + '</p>';
        }
        
        var modal = document.getElementById('evaluationDetailModal');
        if (modal) modal.classList.add('active');
    })
    .catch(function(error) {
        alert('❌ 加载失败：' + error.message);
    });
}

function closeEvaluationDetail() {
    var modal = document.getElementById('evaluationDetailModal');
    if (modal) modal.classList.remove('active');
}

function loadEvaluationMetrics() {
    // 指标在 loadEvaluationReports 中更新
}

function deleteQuestion(id) {
    alert('删除功能开发中...');
}