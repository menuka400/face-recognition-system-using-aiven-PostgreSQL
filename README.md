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

## 📸 what is the logic behind this
1. Vector representation, embeddings and search.

Information can be stored in several ways, just think about the sentence `I Love Parks`:
you could represent it in a table with three columns to flag the presence or not of each word (`I`, `LOVE` and `PARKS`) as per image below:

![image](https://github.com/user-attachments/assets/c57920ce-88d1-4ca8-9242-31347b45c131)


This is a lossless method, no information (apart from the order of words) is lost with this 
encoding. The drawback though is that the number of columns grows with the number of distinct words within the sentence. For example, if we try to also encode
`I Love Croissants` with the same structure we'll end up with four columns `I`, `LOVE`, `PARKS` and `CROISSANTS` as shown below.

![image](https://github.com/user-attachments/assets/83884019-2e78-4aa9-baab-475cffc9f8f1)

2. Embeddings

What are embeddings then? As mentioned above, storing the presence of each word in a 
separate column would create a very wide and unmanageable dataset. Therefore a standard 
approach is to try to reduce the dimensionality by aggregating or dropping some of the 
redundant or not very distiguishable information. In our previous example, we could still 
encode the same information by:

i. dropping the `I` column since it doesn't add any value (it's always 1)

ii. dropping the `CROISSANTS` column since we can still distinguish the two sentences by the presence of the `PARK` word.

If we visualize the two sentences above in a graph only using the `LOVE` and `PARKS` axis 
(therefore excluding the `I` and `CROISSANTS`), the result shows that `I Love Parks` is 
encoded as `(1,1)` since it has present both the `LOVE` and the `PARKS` words. On the other 
hand `I Love Croissants` is encoded with `(1,0)` since it includes `LOVE` but not `PARKS`.

![image](https://github.com/user-attachments/assets/187b2f5a-e969-4e99-bd1c-abd5fb1fd654)

In the graph above, the `distance` represents a calculation of similarity between two vectors: 
The more two vectors point to the same direction or are close to each other, the more the 
information they represent should be similar.

3. Does this work with pictures?

A similar approach also works for pictures. As beautifully explained by [Mathias Grønne](https://towardsdatascience.com/introduction-to-image-embedding-and-accuracy-53473e8965f/) and 
visualized in the image below (it's Figure 1.1 from the above blog, original book photo photo by 
[Jess Bailey](https://unsplash.com/photos/gL2jT6xHYOY) on [Unsplash](https://unsplash.com/)), an image is just a series of characters in a matrix, and therefore we 
could reduce the matrix information and create embeddings on it.

![image](https://github.com/user-attachments/assets/be353354-439e-42ff-929f-157d8bf782af)

4. Setup Face recognition with Python and PostgreSQL `pgvector`

If you, like me, use IPhotos on Mac, you’ll be familiar with the “People” tab, where you can 
select one person and find the photos where this person is included. I used the following 
code to do the same sort of thing with the pictures coming from Crab Week - you’re invited to 
run it, with adaptations, on top of any folder containing images.

Since images are sensitive data, we don't want to rely on any online service or upload them to 
the internet. The entire pipeline defined below is working 100% locally.

The data pipeline will involve several steps:

i. Download all the pictures in a local folder
ii. Retrieve the faces included in any picture
iii. Calculate the embeddings from the faces
iv. Store the embedidngs in PostgreSQL in a `vector` column from `pgvector`
v. Get a colleague picture from Slack
vi. Identify the face in the picture (needed since people can have all types of pictures in Slack)
vii. Calculate the embeddings in the Slack picture
viii. Use `pgvector` distance function to retrieve the closest faces and therefore photos

The entire flow is shown in the picture below:

![image](https://github.com/user-attachments/assets/ff75acc8-6be1-4020-98a1-3bfb788e66ac)

![image](https://github.com/user-attachments/assets/c77a1a43-a10d-417e-8dbf-cded72df460d)





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
3. **Create a PostgreSQL service** (you can select the free plan) and skip all other configurations.
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

CREATE TABLE pictures (picture text PRIMARY KEY,embedding vector(768));
```
5. **Update the `<SERVICE_URI>`** in your `face_recognition.py` code with your Aiven service URI.
   
![image](https://github.com/user-attachments/assets/e83297d5-01b7-4a56-865f-d66b71408aed)


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

