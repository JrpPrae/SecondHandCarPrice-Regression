# Second-hand Car Price Regression

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Neural%20Network-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Linear%20Regression-F7931E?logo=scikitlearn&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-3.35.0-FF7C00?logo=gradio&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?logo=googlecolab&logoColor=white)

โปรแกรม **ทำนายราคารถยนต์มือสอง** ในประเทศไทยด้วยโมเดล **Regression** จากข้อมูลรถมือสองจากเว็บไซต์ **Rod Kaidee** เช่น ยี่ห้อ รุ่น ขนาดเครื่องยนต์ ปี เลขไมล์ จังหวัด และสี เปรียบเทียบโมเดล 3 แบบ ได้แก่ **Linear Regression (scikit-learn)**, **Neural Network 1 ชั้น** และ **Neural Network 2 ชั้น (MLP) ด้วย PyTorch** แล้วนำโมเดลที่ดีที่สุดไปทำเว็บแอปด้วย **Gradio** ให้ผู้ใช้เลือกข้อมูลรถแล้วดูราคาที่ทำนายเป็นบาท

## สารบัญ

- [ภาพรวม](#ภาพรวม)
- [Workflow](#workflow)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [โมเดล](#โมเดล)
- [การเทรนโมเดล](#การเทรนโมเดล)
- [ผลการทดลอง](#ผลการทดลอง)
- [Gradio Web App](#gradio-web-app)
- [Tech Stack](#tech-stack)
- [โครงสร้างโปรเจกต์](#โครงสร้างโปรเจกต์)
- [วิธีใช้งาน](#วิธีใช้งาน)
- [หมายเหตุและสิ่งที่ควรพัฒนาต่อ](#หมายเหตุและสิ่งที่ควรพัฒนาต่อ)

## ภาพรวม

| Notebook | เนื้อหา |
| --- | --- |
| [`SecondHandCarPrice_Prediction_Regression.ipynb`](SecondHandCarPrice_Prediction_Regression.ipynb) | Notebook หลัก: โหลดและทำความสะอาดข้อมูล → One-hot Encoding + MinMaxScaler → เทรนโมเดล 3 แบบ → ประเมินผลด้วย MSE → บันทึกโมเดล → ทำนายข้อมูลใหม่ → สร้างแอป Gradio |
| [`SecondHandCarPrice_gradio.ipynb`](SecondHandCarPrice_gradio.ipynb) | เฉพาะส่วนเว็บแอป Gradio (ต้องมีไฟล์ในโฟลเดอร์ `trained_models/` จาก notebook หลัก) |

**ประเภทงาน:** Supervised Learning — Regression (ทำนายค่าต่อเนื่อง คือราคาเป็นบาท)

## Workflow

```text
┌──────────────── SecondHandCarPrice_Prediction_Regression.ipynb ────────────────┐
│                                                                                │
│  1. ดาวน์โหลด Dataset                                                            │
│     git clone biodatlab/deep-learning-skooldio → rodkaidee-dataset/            │
│     Rodkaidee Data.csv (7,396 แถว × 9 คอลัมน์)                                   │
│              │                                                                 │
│              ▼                                                                 │
│  2. Data Cleaning                                                              │
│     Mileage, Price: "15,640" → 15640.0 (ลบ comma แปลงเป็น float)                  │
│              │                                                                 │
│              ▼                                                                 │
│  3. แบ่งข้อมูล Train / Validation = 80 : 20 (random_state=42)                     │
│              │                                                                 │
│              ▼                                                                 │
│  4. Feature Engineering (fit กับ train set เท่านั้น)                                │
│     - Categorical 6 คอลัมน์ → OneHotEncoder → 136 features                       │
│     - Year, Mileage → MinMaxScaler → 2 features                                │
│     - Price (target) → MinMaxScaler → 0–1                                      │
│     รวมเป็น X: 138 features , y: 1 ค่า                                            │
│     บันทึก encoder/scaler ด้วย joblib → trained_models/                           │
│              │                                                                 │
│              ▼                                                                 │
│  5. เทรนและเปรียบเทียบโมเดล (วัดด้วย MSE บน validation set)                          │
│     ├─ Linear Regression (scikit-learn)                                        │
│     ├─ Neural Network 1 ชั้น (PyTorch)   ← เทียบเท่า Linear Regression             │
│     └─ Neural Network 2 ชั้น / MLP       ← ดีที่สุด                                │
│              │                                                                 │
│              ▼                                                                 │
│  6. บันทึกโมเดล (.joblib / .pth) และทดสอบทำนายราคาจริงเป็นบาท                          │
│     price_scaler.inverse_transform(y_pred)                                     │
└──────────────┬─────────────────────────────────────────────────────────────────┘
               ▼
┌──────────────────────── SecondHandCarPrice_gradio.ipynb ───────────────────────┐
│  7. โหลด MLP + OneHotEncoder + Scalers                                          │
│  8. ผู้ใช้เลือก Brand, Model, Engine, Segment, Province, Color, Year, Mileage      │
│  9. สร้าง DataFrame 1 แถว → transform → model → inverse_transform               │
│ 10. แสดงราคาที่ทำนาย (บาท)                                                        │
└────────────────────────────────────────────────────────────────────────────────┘
```

## Dataset

- **ที่มา:** `rodkaidee-dataset/Rodkaidee Data.csv` จาก repository [biodatlab/deep-learning-skooldio](https://github.com/biodatlab/deep-learning-skooldio) (ข้อมูลประกาศขายรถมือสองจาก Rod Kaidee)
- **ขนาด:** 7,396 แถว × 9 คอลัมน์
- **ยี่ห้อ:** 6 ยี่ห้อ — Toyota, Honda, Mazda, Mitsubishi, Nissan, Suzuki (27 รุ่น)

| คอลัมน์ | ชนิด | ตัวอย่าง | การใช้งาน |
| --- | --- | --- | --- |
| `Brand` | Categorical | Mazda | Feature (One-hot) |
| `Model` | Categorical | Mazda2 | Feature (One-hot) |
| `Engine` | Categorical | 1.3 | Feature (One-hot) |
| `Segment` | Categorical | B-Segment, C-Segment, D-Segment, Eco Car | Feature (One-hot) |
| `Province` | Categorical | กรุงเทพมหานคร | Feature (One-hot) |
| `Color` | Categorical | Gray | Feature (One-hot) |
| `Year` | Numerical | 2022 | Feature (MinMaxScaler) |
| `Mileage` | Numerical (string มี comma) | "15,640" | Feature (ทำความสะอาด + MinMaxScaler) |
| `Price` | Numerical (string มี comma) | "499,000" | **Target** (ทำความสะอาด + MinMaxScaler) |

| ชุดข้อมูล | จำนวน | สัดส่วน |
| --- | --- | --- |
| Train | 5,916 | 80% |
| Validation | 1,480 | 20% |

## Data Preprocessing

```python
CAT_COLUMN = ["Brand", "Model", "Engine", "Segment", "Province", "Color"]

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
year_scaler, mileage_scaler, price_scaler = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()

x_train = np.hstack([
    ohe.fit_transform(df_train[CAT_COLUMN]),            # 136 features
    year_scaler.fit_transform(df_train[["Year"]]),      # 1 feature
    mileage_scaler.fit_transform(df_train[["Mileage"]]) # 1 feature
])                                                      # รวม 138 features
y_train = price_scaler.fit_transform(df_train[["Price"]])
```

- **One-hot Encoding** แปลงข้อมูลประเภทหมวดหมู่เป็นเวกเตอร์ 0/1 ใช้ `handle_unknown="ignore"` เพื่อให้ค่าที่ไม่เคยเห็นตอนเทรนกลายเป็นศูนย์ทั้งหมดแทนการ error
- **MinMaxScaler** ปรับค่าตัวเลขให้อยู่ในช่วง 0–1 เพื่อไม่ให้ฟีเจอร์ที่มีค่าสูง (เช่น เลขไมล์หลักแสน) มีน้ำหนักมากเกินไป และช่วยให้ Neural Network เทรนได้เสถียร
- **fit เฉพาะ train set** แล้วใช้ `transform` กับ validation set เพื่อป้องกัน data leakage
- บันทึก encoder และ scaler ทั้ง 4 ตัวด้วย `joblib` เพื่อใช้แปลงข้อมูลใหม่ตอนทำนาย และใช้ `price_scaler.inverse_transform()` แปลงผลลัพธ์กลับเป็นบาท

## โมเดล

| โมเดล | Framework | โครงสร้าง | ไฟล์ที่บันทึก |
| --- | --- | --- | --- |
| Linear Regression | scikit-learn | `LinearRegression()` | `carprice_model_linear_regression.joblib` |
| `CarPriceModel` (1 ชั้น) | PyTorch | `Linear(138 → 1)` — เทียบเท่า Linear Regression | `carprice_model_mse_0003.pth` |
| `CarPriceTwoLayerModel` (MLP) | PyTorch | `Linear(138 → 10)` → `ReLU` → `Linear(10 → 1)` | `carprice_two_layer_model_mse_00015.pth` |

MLP มี hidden layer ขนาด 10 และ activation แบบ ReLU ทำให้เรียนรู้ความสัมพันธ์แบบไม่เป็นเส้นตรง (non-linear) ระหว่างฟีเจอร์กับราคาได้ เช่น ราคาที่ลดลงไม่เท่ากันในแต่ละช่วงอายุรถ

## การเทรนโมเดล

| Hyperparameter | ค่า |
| --- | --- |
| Loss Function | `nn.MSELoss(reduction="mean")` |
| Optimizer | `torch.optim.Adam` |
| Learning Rate | `1e-3` |
| Batch Size | 32 (train: `shuffle=True`, validation: `shuffle=False`) |
| Epochs | 120 |
| Dataset | `CarPriceDataset` (custom `torch.utils.data.Dataset` จาก numpy array รองรับทั้งมีและไม่มี label) |

```python
for x, y in train_dataloader:
    y_pred = model(x.float())
    loss = loss_fn(y_pred, y.float())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## ผลการทดลอง

ประเมินด้วย **Mean Squared Error (MSE)** บน validation set (1,480 คัน) โดยราคาอยู่ในสเกล 0–1 (MinMaxScaler) ยิ่งต่ำยิ่งดี

| โมเดล | MSE (scaled) |
| --- | --- |
| Linear Regression (scikit-learn) | 0.003113 |
| Neural Network 1 ชั้น (PyTorch) | 0.003121 |
| **Neural Network 2 ชั้น (MLP)** | **0.001879** ✅ |

### ข้อสังเกต

- Linear Regression กับ Neural Network 1 ชั้นได้ MSE ใกล้เคียงกันมาก ยืนยันว่า NN 1 ชั้นที่ไม่มี activation ก็คือ Linear Regression
- MLP ที่มี hidden layer + ReLU ลด MSE ลงได้ประมาณ **40%** เพราะจับความสัมพันธ์แบบไม่เป็นเส้นตรงได้ จึงเลือกไปใช้ในแอป
- มีกราฟ **Predictions vs Actual** ของแต่ละโมเดล จุดที่อยู่ใกล้เส้นทแยง `y = x` แปลว่าทำนายได้ใกล้ราคาจริง

ตัวอย่างการทำนายของ MLP บน validation set:

| Brand | Model | Year | Mileage | Province | ราคาจริง (บาท) | ราคาทำนาย (บาท) |
| --- | --- | --- | --- | --- | --- | --- |
| Toyota | Vios 1.5 | 2015 | 62,639 | กรุงเทพมหานคร | 338,000 | 352,310 |
| Toyota | Altis 1.6 | 2015 | 65,615 | สงขลา | 385,000 | 437,751 |
| Honda | Civic 2.0 | 2007 | 203,000 | สิงห์บุรี | 295,000 | 301,238 |
| Toyota | Yaris 1.2 | 2014 | 124,007 | ราชบุรี | 339,000 | 336,619 |
| Mazda | Mazda2 1.3 | 2019 | 51,957 | กรุงเทพมหานคร | 439,000 | 428,297 |

## Gradio Web App

```python
gr.Interface(
    fn=predict_car_price,
    inputs=[
        gr.Dropdown(brands,    label="Brand"),
        gr.Dropdown(models,    label="Model"),
        gr.Dropdown(engines,   label="Engine Size"),
        gr.Dropdown(segments,  label="Car segment"),
        gr.Dropdown(provinces, label="Province"),
        gr.Dropdown(colors,    label="Color"),
        gr.Slider(1990, 2023,  label="Year"),
        gr.Slider(0, 400000,   label="Mileage"),
    ],
    outputs=gr.Textbox(label="ราคาทำนาย (บาท)"),
    examples=examples,          # ตัวอย่างข้อมูลรถ 10 คัน กดเพื่อเติมค่าอัตโนมัติ
    title="ทำนายราคารถมือสอง",
)
```

| Input | ตัวเลือก |
| --- | --- |
| Brand | 6 ยี่ห้อ |
| Model | 27 รุ่น |
| Engine Size | 1.0 – 3.0 ลิตร (13 ขนาด) |
| Car segment | B-Segment, C-Segment, D-Segment, Eco Car |
| Province | 74 จังหวัด |
| Color | 12 สี |
| Year | 1990 – 2023 (Slider) |
| Mileage | 0 – 400,000 กม. (Slider) |

ขั้นตอนใน `predict_car_price()`: สร้าง DataFrame 1 แถว → `ohe.transform` + `year_scaler` + `mileage_scaler` → ส่งเข้า MLP ผ่าน `DataLoader` → `price_scaler.inverse_transform` → คืนราคาเป็นจำนวนเต็ม (บาท)

## Tech Stack

| หมวด | เทคโนโลยี | หน้าที่ในโปรเจกต์ |
| --- | --- | --- |
| ภาษา | **Python** | ภาษาหลัก |
| Data Manipulation | **pandas**, **NumPy** | โหลด CSV ทำความสะอาดข้อมูล และรวม feature matrix |
| Machine Learning | **scikit-learn** | `train_test_split`, `OneHotEncoder`, `MinMaxScaler`, `LinearRegression`, `mean_squared_error` |
| Deep Learning | **PyTorch** (`torch.nn`, `Dataset`, `DataLoader`) | สร้างและเทรน Neural Network |
| Model Persistence | **joblib**, `torch.save` | บันทึก encoder/scaler/Linear Regression และ weights ของ NN |
| Visualization | **Matplotlib** | กราฟ Predictions vs Actual |
| Utilities | **tqdm** | progress bar |
| Web UI | **Gradio 3.35.0** (`gr.Interface`) | เว็บแอปทำนายราคา |
| Environment | **Google Colab** | รัน notebook |

## โครงสร้างโปรเจกต์

```text
SecondHandCarPrice-Regression/
├── SecondHandCarPrice_Prediction_Regression.ipynb   # เตรียมข้อมูล, เทรน 3 โมเดล, ประเมินผล, บันทึก, Gradio
├── SecondHandCarPrice_gradio.ipynb                  # เฉพาะเว็บแอป Gradio
└── README.md
```

ไฟล์ที่ถูกสร้างขึ้นระหว่างรัน notebook (ไม่ได้อยู่ใน repository):

```text
deep-learning-skooldio/                          # dataset ที่ clone มา
trained_models/
├── one_hot_encoder.joblib
├── year_scaler.joblib
├── mileage_scaler.joblib
├── price_scaler.joblib
├── carprice_model_linear_regression.joblib      # Linear Regression
├── carprice_model_mse_0003.pth                  # Neural Network 1 ชั้น
└── carprice_two_layer_model_mse_00015.pth       # MLP 2 ชั้น (ใช้ในแอป)
```

## วิธีใช้งาน

### ตัวเลือกที่ 1: Google Colab (แนะนำ)

1. เปิด [Google Colab](https://colab.research.google.com/) → **File → Upload notebook** → เลือก `SecondHandCarPrice_Prediction_Regression.ipynb`
2. กด **Runtime → Run all** — notebook จะ clone dataset เทรนโมเดล บันทึกไฟล์ใน `trained_models/` และเปิดแอป Gradio พร้อม public URL
3. ถ้าต้องการรันเฉพาะแอป ให้เปิด `SecondHandCarPrice_gradio.ipynb` ใน session เดียวกัน (หรืออัปโหลดโฟลเดอร์ `trained_models/` ขึ้นไปก่อน)

### ตัวเลือกที่ 2: รันบนเครื่องตัวเอง

```bash
pip install pandas numpy scikit-learn torch matplotlib tqdm joblib "gradio==3.35.0" jupyter
git clone https://github.com/biodatlab/deep-learning-skooldio
jupyter notebook SecondHandCarPrice_Prediction_Regression.ipynb
```

> คำสั่งที่ขึ้นต้นด้วย `!` ใน notebook (`!git clone`, `!pip install`) เป็นคำสั่งของ Jupyter/Colab ถ้า clone dataset เองแล้วข้าม cell นั้นได้

## หมายเหตุและสิ่งที่ควรพัฒนาต่อ

### ข้อควรระวังในโค้ด

- **ไฟล์ใน `trained_models/` ไม่ได้อยู่ใน repository** ต้องรัน notebook หลักก่อน ไม่อย่างนั้น `SecondHandCarPrice_gradio.ipynb` จะหาไฟล์ไม่เจอ
- **ชื่อไฟล์โมเดลไม่ตรงกับผลจริง** — `carprice_two_layer_model_mse_00015.pth` แต่ MSE ที่ได้คือ 0.00188
- **Validation loss ระหว่างเทรนเป็นค่าของ batch สุดท้าย** — ใน `train()` ค่า `loss` ที่พิมพ์ออกมาเป็นของ batch สุดท้ายของ validation set ไม่ใช่ค่าเฉลี่ยทั้งชุด ควรสะสมแล้วหารด้วยจำนวน batch
- **กราฟของ MLP สลับแกน** — cell ที่พล็อตผลของ MLP ใส่ค่าจริงไว้แกน X แต่ตั้งชื่อแกนว่า "Predictions value"
- **แอปไม่กรองรุ่นตามยี่ห้อ** — ผู้ใช้เลือกคู่ที่ไม่มีจริงได้ (เช่น Toyota + Civic) และเพราะ `handle_unknown="ignore"` โมเดลจะยังทำนายราคาออกมาโดยไม่แจ้งเตือน ควรทำ Dropdown รุ่นที่เปลี่ยนตามยี่ห้อ และกรองขนาดเครื่องยนต์และ Segment ตามรุ่น
- **ผลลัพธ์ไม่ได้จัดรูปแบบตัวเลข** — แสดงเป็น `352310` แทน `352,310 บาท` ตามที่ placeholder บอกไว้
- **Gradio 3.35.0** เป็นเวอร์ชันเก่า ถ้าอัปเกรดเป็น Gradio 4 ขึ้นไป บาง argument ต้องปรับ
- ใน notebook หลักมี cell `!pip install` ที่ไม่มีชื่อแพ็กเกจ ซึ่งจะ error ควรลบออก

### แนวทางพัฒนาต่อ

- แยก **Test set** ออกจาก Validation set เพื่อรายงานผลที่ไม่ลำเอียง
- รายงานค่า error ในหน่วยบาท เช่น **RMSE** และ **MAE** และ **R²** ซึ่งตีความได้ง่ายกว่า MSE ในสเกล 0–1
- ทดลองโมเดลแบบ Tree-based เช่น **Random Forest**, **XGBoost**, **LightGBM** ซึ่งมักทำงานได้ดีกับข้อมูลตาราง
- ใช้ **Embedding layer** สำหรับฟีเจอร์ที่มีหลายค่า เช่น Province แทน One-hot
- เพิ่มฟีเจอร์ **อายุรถ** (ปีปัจจุบัน − Year) และทำ **log transform** กับราคาและเลขไมล์
- ทำ Early Stopping และบันทึกโมเดลที่มี validation loss ต่ำที่สุด
- Deploy แอปบน **Hugging Face Spaces** เพื่อให้ใช้งานออนไลน์ได้ถาวร
