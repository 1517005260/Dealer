# Dealer 后端服务

基于 Django 的后端 API 服务，实现 Dealer 模型市场的核心算法和数据管理功能。

## 系统架构

### 核心模块

1. **数据管理模块**：管理训练和测试数据集（Cancer、Chess、Iris）
2. **Shapley 值计算模块**：实现蒙特卡洛 Shapley 值估算
3. **差分隐私训练模块**：实现带噪声的隐私保护模型训练
4. **定价算法模块**：实现收入最大化和无套利定价
5. **模型管理模块**：管理模型版本和发布状态

### 数据模型

- `TrainCancer/TestCancer`：乳腺癌诊断数据集
- `TrainChess/TestChess`：国际象棋终局数据集  
- `TrainIris/TestIris`：鸢尾花分类数据集
- `ModelInfo`：模型版本信息
- `ShapleyInfo`：Shapley 值计算结果
- `SurveyInfo`：市场调研价格信息

## 环境要求

- Python 3.7 或更高版本
- Django 3.0+
- 相关 Python 包（见 requirements.txt）

## 安装步骤

### 1. 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv dealer_env

# 激活虚拟环境
# Windows:
dealer_env\Scripts\activate
# Linux/Mac:
source dealer_env/bin/activate
```

### 2. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 3. 数据库初始化

```bash
# 创建数据库迁移文件
python manage.py makemigrations

# 执行数据库迁移
python manage.py migrate
```

### 4. 启动开发服务器

```bash
python manage.py runserver
```

服务器将在 `http://localhost:8000` 启动。

## API 接口说明

### 数据查询接口

#### 获取数据集
- `GET /query_cancer/` - 获取癌症数据集
- `GET /query_chess/` - 获取国际象棋数据集
- `GET /query_iris/` - 获取鸢尾花数据集

#### 按 ID 查询数据
- `POST /query_cancer_by_id/` - 按 ID 查询癌症数据
- `POST /query_chess_by_id/` - 按 ID 查询国际象棋数据

### 核心算法接口

#### Shapley 值计算
```bash
POST /query_compensation/
Content-Type: application/json

{
    "dataset": "cancer",
    "id": [1, 2, 3, 4, 5],
    "bp": 1.0,
    "ps": 0.5,
    "eps": 1.0,
    "sample": 100
}
```

**参数说明**：
- `dataset`: 数据集名称 (cancer/chess/iris)
- `id`: 参与计算的数据 ID 列表
- `bp`: 基础价格参数
- `ps`: 隐私敏感性参数
- `eps`: 差分隐私参数 ε
- `sample`: 蒙特卡洛采样次数

#### 差分隐私模型训练
```bash
POST /query_amp_shapley/
Content-Type: application/json

{
    "dataset": "cancer",
    "num_repeats": 10,
    "shapley_mode": "full",
    "epsilon": [1.0, 2.0, 3.0],
    "price": [100, 200, 300],
    "budget": 1000,
    "bp": 1.0,
    "ps": 0.5
}
```

**参数说明**：
- `num_repeats`: 训练重复次数
- `shapley_mode`: Shapley 计算模式
- `epsilon`: 隐私预算列表
- `price`: 对应价格列表
- `budget`: 总预算限制

### 模型管理接口

#### 发布模型
```bash
POST /release_model/
Content-Type: application/json

{
    "id": 1
}
```

#### 查询可用模型
```bash
GET /query_all_model/
```

#### 按条件查询模型
```bash
POST /query_limited_model/
Content-Type: application/json

{
    "dataset": "cancer",
    "budget": 500,
    "covexp": 0.8,
    "covsen": 1.0,
    "noiexp": 1.0,
    "noisen": 0.5
}
```

### 定价算法接口

#### 市场调研和定价
```bash
POST /write_survey/
Content-Type: application/json

{
    "survey": [
        {"eps": 1.0, "pri": 100},
        {"eps": 2.0, "pri": 200},
        {"eps": 3.0, "pri": 300}
    ]
}
```

## 开发指南

### 添加新数据集

1. 在 `models.py` 中定义新的数据模型：
```python
class TrainNewDataset(models.Model):
    id = models.IntegerField(null=False, primary_key=True)
    feature1 = models.FloatField(null=False)
    feature2 = models.FloatField(null=False)
    # ... 更多特征
    label = models.IntegerField(null=False)
```

2. 创建迁移文件并执行：
```bash
python manage.py makemigrations
python manage.py migrate
```

3. 在 `views.py` 中添加相应的查询接口。

### 修改算法参数

核心算法实现在 `dealer/utils/` 目录下：
- `AMP.py`: 基础模型训练
- `AMP_shapley.py`: Shapley 值相关计算
- `Price.py`: 定价算法
- `Gen_Shapley.py`: Shapley 值生成
- `Draw.py`: 可视化工具

### 自定义效用函数

在 `Gen_Shapley.py` 中修改 `eval_monte_carlo` 函数，自定义模型评估指标。

## 测试

```bash
# 运行所有测试
python manage.py test
```

## 部署

### 开发环境部署
```bash
# 设置环境变量
export DJANGO_SETTINGS_MODULE=dealer_demo.settings
export PYTHONPATH=$PYTHONPATH:/path/to/dealer

# 启动服务
python manage.py runserver 0.0.0.0:8000
```

```bash
pip install -r requirement.txt
python manage.py makemigrations
python manage.py migrate
nohup python manage.py runserver 0.0.0.0:8000 >> /root/stat.log 2>&1 &
```

