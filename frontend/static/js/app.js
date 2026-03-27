// frontend/static/js/app.js

// ==================== 全局配置 ====================
const CONFIG = {
    apiUrl: 'http://localhost:8000',  // 支持代理，相对路径
    apiKey: '',
    timeout: 60000
};

// 配置 marked 渲染选项
if (typeof marked !== 'undefined') {
    marked.setOptions({
        breaks: true,           // 支持 \n 换行
        gfm: true,              // 支持 GitHub Flavored Markdown
        headerIds: false,       // 不生成 header id（避免冲突）
        mangle: false,          // 不转义邮箱
        sanitize: false         // 不自动过滤 HTML（医疗内容可信）
    });
}
// 对话历史
let chatHistory = [];

// ==================== 初始化 ====================
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    checkHealth();
    loadSettings();
    
    // 定期健康检查
    setInterval(checkHealth, 300000);
});

// ==================== 导航 ====================
function initNavigation() {
    const menuItems = document.querySelectorAll('.menu-item');
    menuItems.forEach(item => {
        item.addEventListener('click', () => {
            // 更新菜单状态
            menuItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            
            // 切换页面
            const page = item.dataset.page;
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.getElementById(`${page}Page`).classList.add('active');
            
            // 页面特定初始化
            if (page === 'stats') {
                loadStats();
            }
            if (page === 'evaluation') {
                loadEvaluationPage();
            }
        });
    });
}

// ==================== 健康检查 ====================
async function checkHealth() {
    try {
        const response = await fetch(`${CONFIG.apiUrl}/health`);
        const data = await response.json();
        
        const indicator = document.getElementById('systemStatus');
        const text = document.getElementById('statusText');
        
        if (data.status === 'healthy') {
            indicator.classList.add('online');
            text.textContent = '系统正常';
            text.style.color = '#27ae60';
        } else {
            indicator.classList.remove('online');
            text.textContent = '部分服务异常';
            text.style.color = '#f39c12';
        }
    } catch (error) {
        const indicator = document.getElementById('systemStatus');
        const text = document.getElementById('statusText');
        indicator.classList.remove('online');
        text.textContent = '连接失败';
        text.style.color = '#e74c3c';
    }
}

// ==================== 发送消息 ====================

async function sendMessage() {
    const input = document.getElementById('queryInput');
    const query = input.value.trim();
    
    if (!query) return;
    
    // ✅ 修复：添加默认值，防止元素不存在时报错
    const modeElement = document.getElementById('retrievalMode');
    const topKElement = document.getElementById('topK');
    
    const mode = modeElement ? modeElement.value : 'hybrid_rerank';
    const topK = topKElement ? parseInt(topKElement.value) : 5;
    
    // 禁用输入
    setInputEnabled(false);
    
    // 添加用户消息
    addMessage(query, 'user');
    input.value = '';
    
    // 添加到历史
    chatHistory.push({ role: 'human', content: query });
    
    try {
        // 创建助手消息占位
        const assistantMessage = addMessage('思考中...', 'assistant', true);
        const contentDiv = assistantMessage.querySelector('.message-content');
        
        // ✅ 修复：使用正确的 API 地址
        const apiUrl = CONFIG.apiUrl.replace(/\/$/, ''); // 去掉末尾斜杠
        
        // 流式请求
        const response = await fetch(`${apiUrl}/chat/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': CONFIG.apiKey || ''
            },
            body: JSON.stringify({
                query: query,
                chat_history: chatHistory.slice(-10),
                retrieval_mode: mode,
                top_k: topK
                // ✅ 不传 filter，避免空对象报错
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullAnswer = '';
        let citations = [];
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') continue;
                    
                    try {
                        const event = JSON.parse(data);
                        
                        if (event.type === 'chunk') {
                            fullAnswer += event.content;
                            // ✅ 流式时显示纯文本 + 光标
                            contentDiv.textContent = fullAnswer + '▊';
                            const container = document.getElementById('chatContainer');
                            container.scrollTop = container.scrollHeight;
                            
                        } else if (event.type === 'complete') {
                            // ✅ 完整后用 marked 渲染 Markdown
                            const rendered = typeof marked !== 'undefined' 
                                ? marked.parse(fullAnswer) 
                                : fullAnswer.replace(/\n/g, '<br>');
                            contentDiv.innerHTML = rendered;
                            
                            citations = event.citations || [];
                            if (citations.length > 0) {
                                addCitations(contentDiv, citations);
                            }
                            
                            chatHistory.push({ role: 'assistant', content: fullAnswer });
                            
                        } else if (event.type === 'retrieval_complete') {
                            contentDiv.textContent = `已检索 ${event.document_count} 条文献，正在生成回答...\n\n`;
                            
                        } else if (event.error) {
                            throw new Error(event.error);
                        }
                    } catch (e) {
                        console.error('解析错误:', e);
                    }
                }
            }
        }
        
    } catch (error) {
        console.error('发送失败:', error);
        addMessage(`❌ 错误：${error.message}`, 'error');
        chatHistory.pop();
    } finally {
        setInputEnabled(true);
    }
}

// ==================== 辅助函数 ====================

function addMessage(content, type, isLoading = false) {
    const container = document.getElementById('chatContainer');
    
    // 移除欢迎消息
    const welcome = container.querySelector('.welcome-message');
    if (welcome) welcome.remove();
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    // ✅ 关键修复：用 marked 渲染 Markdown
    const renderedContent = typeof marked !== 'undefined' 
        ? marked.parse(content) 
        : content.replace(/\n/g, '<br>');  // 降级方案
    
    messageDiv.innerHTML = `
        <div class="message-content markdown-body">${renderedContent}</div>
        ${isLoading ? '<div class="loading-indicator">⏳</div>' : ''}
    `;
    
    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;
    
    return messageDiv;
}

function addCitations(container, citations) {
    const citationsDiv = document.createElement('div');
    citationsDiv.className = 'citations';
    citationsDiv.innerHTML = '<strong>📚 参考文献：</strong>';
    
    citations.forEach((cite, i) => {
        const item = document.createElement('div');
        item.className = 'citation-item';
        item.innerHTML = `
            <strong>[${cite.index}]</strong> ${cite.source}
            ${cite.page ? `(第${cite.page}页)` : ''}
            ${cite.score ? ` (相关度：${(cite.score * 100).toFixed(1)}%)` : ''}
        `;
        citationsDiv.appendChild(item);
    });
    
    container.appendChild(citationsDiv);
}

function setInputEnabled(enabled) {
    const input = document.getElementById('queryInput');
    const btn = document.getElementById('sendBtn');
    input.disabled = !enabled;
    btn.disabled = !enabled;
    btn.querySelector('span:first-child').style.display = enabled ? 'inline' : 'none';
    btn.querySelector('.loading').style.display = enabled ? 'none' : 'inline';
}

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function sendExample(question) {
    const input = document.getElementById('queryInput');
    if (input) {
        input.value = question;
        sendMessage();
    }
}

// ==================== 设置管理 ====================
function loadSettings() {
    const saved = localStorage.getItem('medical_rag_settings');
    if (saved) {
        const settings = JSON.parse(saved);
        CONFIG.apiUrl = settings.apiUrl || '/api';
        CONFIG.apiKey = settings.apiKey || '';
        document.getElementById('apiUrl').value = CONFIG.apiUrl;
        document.getElementById('apiKey').value = CONFIG.apiKey;
    }
}

function saveSettings() {
    CONFIG.apiUrl = document.getElementById('apiUrl').value;
    CONFIG.apiKey = document.getElementById('apiKey').value;
    
    localStorage.setItem('medical_rag_settings', JSON.stringify({
        apiUrl: CONFIG.apiUrl,
        apiKey: CONFIG.apiKey
    }));
    
    alert('设置已保存');
    checkHealth();
}

// ==================== 统计信息 ====================
async function loadStats() {
    try {
        const response = await fetch(`${CONFIG.apiUrl}/collection/stats`);
        const data = await response.json();
        
        const grid = document.getElementById('statsGrid');
        grid.innerHTML = `
            <div class="stat-card">
                <h3>文档总数</h3>
                <div class="value">${data.total_documents}</div>
            </div>
            <div class="stat-card">
                <h3>集合名称</h3>
                <div class="value" style="font-size:18px">${data.collection_name}</div>
            </div>
        `;
    } catch (error) {
        console.error('加载统计失败:', error);
    }
}

// ==================== 文档上传 ====================
async function uploadDocuments() {
    const fileInput = document.getElementById('fileInput');
    const files = fileInput.files;
    
    if (files.length === 0) {
        alert('请选择文件');
        return;
    }
    
    // 这里需要后端支持文件上传接口
    alert('文档上传功能开发中...');
}