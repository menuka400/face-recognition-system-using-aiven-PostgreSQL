# 🚀 Face Recognition System

A powerful and efficient **Face Recognition System** built using OpenCV, imgbeddings, and PostgreSQL. This project enables real-time face detection, registration, and recognition with database storage for embeddings.

## ✨ Features
- 🔍 **Real-time Face Detection** using OpenCV
- 🏷 **User Registration** with automatic embedding storage
- 🤖 **Face Recognition** with high accuracy
- 📊 **Database Management** (View, Edit, and Delete Users)
- 🚀 **PostgreSQL Integration** for scalable storage

## 📸 How It Works
1. Capture a face using your webcam.
2. Generate an embedding using `imgbeddings`.
3. Store the embedding in a PostgreSQL database.
4. Match faces against the stored database for recognition.

## 🛠 Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/face-recognition-system.git
cd face-recognition-system
```

### 2️⃣ Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up PostgreSQL
Follow these steps to set up PostgreSQL:

#### 🏗 Step 1: Create a PostgreSQL Service on Aiven
1. **Create an account** on [Aiven.io](https://aiven.io).
2. **Create a new project** in Aiven.
3. **Create a PostgreSQL service** (you can select the free plan or any other option) and skip all other configurations.
4. **Wait for the server** to transition from *Rebuilding* to *Running*.

#### 🔧 Step 2: Install PostgreSQL Locally
1. Download & install **PostgreSQL** from [EnterpriseDB](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads).
2. Restart your computer after installation (this is required).

#### 🌐 Step 3: Connect to Your PostgreSQL Service
1. Go to **Aiven.io** → **Services**.
2. Select the **PostgreSQL service** you created.
3. Click **Quick Connect** → Choose **PSQL** as the connection method.
4. Copy the provided connection command and **paste it into your terminal**.

#### 📜 Step 4: Initialize the Database
After connecting to PostgreSQL, run the following commands your terminal:

```sql
CREATE EXTENSION vector;

CREATE TABLE pictures (picture TEXT PRIMARY KEY,embedding VECTOR(768));
```
5. **Update the `<SERVICE_URI>`** in your `face_recognition.py` code with your Aiven service URI.

### 4️⃣ Run the Application
```bash
python face_recognition.py
```

## 🎯 Usage
- Select an option from the menu:
  1. **Start Face Recognition**
  2. **Register a New User**
  3. **Manage User Database**
- Follow on-screen instructions to register or recognize faces.

## 📌 Contributing
Want to improve this project? Feel free to fork and contribute! Open an issue for any suggestions.

## 📜 License
This project is licensed under the MIT License.

🚀 Happy Coding! 🎯

