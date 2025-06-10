# Dealer 前端应用

基于 React + Ant Design 的前端用户界面，提供直观的模型市场交互体验。

## 系统架构

### 核心组件

1. **Owner（数据拥有者）**：数据集管理、隐私设置、补偿查看
2. **Broker（代理商）**：模型训练、定价策略、市场管理
3. **Buyer（模型购买者）**：模型浏览、购买决策、需求设置
4. **Shapley**：Shapley 值可视化、数据贡献度展示

### 技术栈

- **React 17**：核心框架
- **Ant Design 4**：UI 组件库
- **ECharts**：数据可视化
- **Axios**：HTTP 请求库
- **React Router**：路由管理

## 环境要求

- Node.js 14.0 或更高版本
- npm 6.0 或 yarn 1.0 或更高版本
- 现代浏览器（Chrome、Firefox、Safari、Edge）

## 安装步骤

### 1. 安装 Node.js 依赖

```bash
cd frontend

# 使用 npm
npm install

# 或使用 yarn
yarn install
```

### 2. 配置后端接口

检查 `src/setupProxy.js` 中的代理配置：

```javascript
const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://localhost:8000',  // 后端服务地址
      changeOrigin: true,
    })
  );
};
```

### 3. 启动开发服务器

```bash
# 使用 npm
npm start

# 或使用 yarn
yarn start
```

应用将在 `http://localhost:3000` 启动，并自动打开浏览器。

## 使用指南

### 界面概览

Dealer 前端界面分为三个主要区域：

```
┌─────────────────────────────────────────────────────────┐
│                    Dealer 模型市场                        │
├─────────────┬─────────────────────┬─────────────────────┤
│   数据拥有者   │        代理商        │     模型购买者      │
│             │                    │                    │
│  • 数据集管理  │   • Shapley 值展示   │   • 模型浏览       │
│  • 隐私设置   │   • 模型训练        │   • 需求设置       │
│  • 补偿查看   │   • 定价策略        │   • 购买决策       │
│             │   • 市场管理        │                    │
└─────────────┴─────────────────────┴─────────────────────┘
```

### 数据拥有者操作流程

#### 1. 选择数据集
- 从下拉菜单中选择可用数据集（Cancer、Chess、Iris）
- 查看数据集基本信息和统计

#### 2. 设置隐私参数
- **基础价格 (bp)**：设置数据的基础价值
- **隐私敏感性 (ps)**：设置对隐私保护的要求程度
  - 值越大，对隐私泄露的补偿要求越高

#### 3. 查看 Shapley 值
- 系统会计算并显示每个数据点的贡献度
- 可视化展示包括：
  - 散点图：显示数据分布和 Shapley 值
  - 柱状图：显示贡献度排名
  - 统计信息：平均值、标准差等

#### 4. 获取补偿信息
- 查看基于 Shapley 值的公平补偿分配
- 了解在不同隐私级别下的补偿金额

### 代理商操作流程

#### 1. Shapley 值分析
- 查看从数据拥有者处收集的 Shapley 值
- 分析数据质量和价值分布
- 决定选择哪些数据用于模型训练

#### 2. 模型训练配置
- **重复次数**：设置训练重复次数以提高稳定性
- **隐私预算 (ε)**：设置差分隐私参数列表
  - 多个 ε 值对应不同的模型版本
  - ε 值越大，隐私保护越弱，但模型准确性越高
- **制造预算**：设置总的模型开发预算

#### 3. 定价策略
- 进行市场调研，收集潜在买家的出价信息
- 系统自动计算无套利定价策略
- 查看完整价格空间和最优收入方案

#### 4. 模型发布管理
- 查看训练完成的模型列表
- 选择发布的模型版本
- 监控模型销售状况

### 模型购买者操作流程

#### 1. 设置购买需求
- **预算**：设置最大购买预算
- **覆盖期望 (θ)**：设置对数据覆盖率的最低要求
- **覆盖敏感性 (δ)**：设置对覆盖率变化的敏感程度
- **噪声期望 (η)**：设置可接受的最大噪声水平
- **噪声敏感性 (γ)**：设置对噪声变化的敏感程度

#### 2. 浏览可用模型
- 查看所有已发布的模型
- 对比不同模型的：
  - 覆盖率（数据质量）
  - 隐私参数（噪声水平）
  - 价格
  - 推荐程度

#### 3. 模型评估和选择
- 系统根据买家需求自动推荐最适合的模型
- 查看模型的详细性能指标
- 比较性价比

#### 4. 购买决策
- 确认选择的模型符合需求和预算
- 完成购买流程

## 开发指南

### 项目结构

```
frontend/
├── public/              # 静态资源
├── src/
│   ├── components/      # React 组件
│   │   ├── Owner/      # 数据拥有者组件
│   │   ├── Broker/     # 代理商组件
│   │   ├── Buyer/      # 模型购买者组件
│   │   └── Shapley/    # Shapley 值可视化组件
│   ├── js/             # JavaScript 工具函数
│   ├── style/          # 样式文件
│   ├── images/         # 图片资源
│   └── App.js          # 主应用组件
├── package.json         # 项目配置
└── README.md           # 文档
```

### 添加新组件

1. 在 `src/components/` 下创建新组件目录
2. 实现组件逻辑：

```javascript
import React, { Component } from 'react';
import { Card, Button } from 'antd';

class NewComponent extends Component {
    state = {
        // 组件状态
    };

    handleAction = () => {
        // 处理用户交互
    };

    render() {
        return (
            <Card title="新组件">
                <Button onClick={this.handleAction}>
                    执行操作
                </Button>
            </Card>
        );
    }
}

export default NewComponent;
```

3. 在父组件中引入和使用

### 修改样式

- 全局样式：修改 `src/App.css`
- 组件样式：在对应组件目录下创建 CSS 文件
- 主题定制：修改 Ant Design 主题变量

### API 集成

使用 Axios 进行 HTTP 请求：

```javascript
import axios from 'axios';

// GET 请求
const fetchData = async () => {
    try {
        const response = await axios.get('/api/query_cancer/');
        return response.data;
    } catch (error) {
        console.error('请求失败:', error);
    }
};

// POST 请求
const postData = async (data) => {
    try {
        const response = await axios.post('/api/query_compensation/', data);
        return response.data;
    } catch (error) {
        console.error('请求失败:', error);
    }
};
```

### 数据可视化

使用 ECharts 创建图表：

```javascript
import ReactECharts from 'echarts-for-react';

const ChartComponent = ({ data }) => {
    const option = {
        title: { text: '数据可视化' },
        xAxis: { data: data.labels },
        yAxis: {},
        series: [{
            type: 'bar',
            data: data.values
        }]
    };

    return <ReactECharts option={option} />;
};
```

## 构建和部署

### 开发构建

```bash
# 构建生产版本
npm run build

# 构建文件将生成在 build/ 目录下
```

### 部署到生产环境

#### 静态文件服务器部署

```bash
# 安装静态文件服务器
npm install -g serve

# 启动服务
serve -s build -l 3000
```

#### Nginx 部署

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        root /path/to/dealer/frontend/build;
        index index.html index.htm;
        try_files $uri $uri/ /index.html;
    }
    
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 环境变量配置

创建 `.env` 文件：

```env
REACT_APP_API_BASE_URL=http://localhost:8000
REACT_APP_TITLE=Dealer 模型市场
```

在代码中使用：

```javascript
const apiUrl = process.env.REACT_APP_API_BASE_URL;
```