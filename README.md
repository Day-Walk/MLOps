# 📖 MLOps Repo

### 🗂️ Github Branch 관리
{BD53A743-450D-4A43-8D1A-BF87268496C3}.png

### 🕹️ Github Issues, Pull requests
프로젝트의 Task 관리

# 🧩 Git Convention
| 커밋 유형 | 의미                     |
| --------- | ------------------------ |
| `feat`      | 기능의 추가              |
| `fix`       | 버그 수정                |
| `docs`      | 주석 추가, README 수정   |
| `chore`     | 환경 설정                |
| `refactor`  | 리팩토링                 |
| `remove`    | 불필요 코드, 파일 제거   |

### ‼️ 규칙에 맞는 좋은 Commit Message를 작성해야 하는 이유
1. 팀원과의 소통
2. 편리하게 과거 추적 가능

### ‼️ 하나의 Commit 에는 한 가지 문제만!
1. 추적 가능하게 유지해주기
2. 너무 많은 문제를 한 Commit에 담으면 추적하기 어려움

# 💻 Code Convention

**1. Indent (들여쓰기)**
- 항상 4개의 공백(space) 사용

👍 좋은 예
```python
def foo():
    for i in range(3):
        print(i)
```
👎 나쁜 예 - **공백 2개**
```python
def foo():
  for i in range(3):
      print(i)
```
**2. Naming Convention (이름 규칙)**
- 변수·함수: snake_case
- 클래스: CamelCase
- 상수: UPPER_CASE_WITH_UNDERSCORES

👍 좋은 예
```python
model_accuracy = 0.9
def train_model():
    pass
class ModelTrainer:
    pass
MAX_EPOCHS = 100
```

👎 나쁜 예
```python
ModelAccuracy = 0.9
def TrainModel():
    pass
class model_trainer:
    pass
maxEpochs = 100
```

**3. Comments (주석)**
- \# 뒤에 한 칸 띄우기
- 핵심 로직에만 간결하게 작성
- 불필요하거나 자명한 주석 금지

👍 좋은 예
```python
# 결측치 제거
data = data.dropna()

def predict(x):
    """모델로 예측값을 반환합니다."""
    return model.predict(x)
```

👎 나쁜 예
```python
#결측치제거
data = data.dropna()  # dropna()

def predict(x):
    """predict"""  # too short
    return model.predict(x)
```

**4. Whitespace Around Operators (기호 앞뒤 공백)**
- 이항 연산자: 양쪽에 한 칸
- 콤마, 콜론 뒤에 한 칸
- 함수 기본값 지정 시 = 양쪽 공백 금지

👍 좋은 예
```python
a = b + 1
foo(x, y=10)
items = [1, 2, 3]
```

👎 나쁜 예
```python
a=b+1
foo(x,y = 10)
items = [1,2,3]
```

**5. Import Rules (Import 규칙)**
- 표준 라이브러리
- 서드파티 라이브러리
- 로컬(프로젝트) 모듈
  - 각 그룹 사이에 빈 줄 한 줄
  - 한 줄에 하나씩 임포트

👍 좋은 예
```python
import os
import sys

import numpy as np
import pandas as pd

from myproject.data.loader import DataLoader
```

👎 나쁜 예
```python
import os, sys
import numpy as np
from myproject.data.loader import DataLoader, ModelTrainer
```
